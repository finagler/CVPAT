"""Script that computes scores obtained by various agents on a set of shoes."""
import collections
import json
import functools
import multiprocessing
import typing

import numpy as np
from numpy import typing as npt
import tqdm

from game import card
from game import hand
from stats import hand_stats
from game import payout_table
from agent import player_agent
from game import poker_game
from game import shoe


# Tuple of seed, list of scores.
ShoeTargetScores = typing.Tuple[int, typing.List[int]]

# Lists of scores, number hands played, and move histories from one or more
# agents playing a single game.
GameInfo = typing.Tuple[typing.List[int], typing.List[int],
                        typing.List[typing.List[poker_game.GameMove]]]


# Enum assigned to each suit in input JSON encoding.
JSON_TO_SUIT = {1: card.SuitEnum.SPADE,
                4: card.SuitEnum.CLUB,
                8: card.SuitEnum.HEART,
                16: card.SuitEnum.DIAMOND}


def _num_to_suit(suit_num: int) -> card.SuitEnum:
    """Returns suit associated with JSON-file suit encoding."""
    return JSON_TO_SUIT[suit_num]


def _num_to_rank(r: int) -> int:
    """Returns rank from JSON-file to rank number with Ace as 1."""
    if r == 14:
        return 1
    if r >= 2 or r <= 13:
        return r
    raise ValueError(f'Invalid rank {r}')


def _json_to_shoe(shoe_data: dict) -> shoe.Shoe:
    """Convert raw shoe data loaded from JSON to shoe.

    Args:
        shoe_data: dict representing a single shoe encoded from raw JSON.
            See analyze_json_data for format details.

    Raises:
        ValueError: number of cards in a deck is not a multiple of 52.

    Returns: list of shoes
    """
    seed = shoe_data['seed']
    deck = shoe_data['deck']
    cards = [card.Card(_num_to_rank(c['rank']), _num_to_suit(c['suit']))
             for c in deck]
    if len(cards) % 52:
        raise ValueError(f'Number of cards in deck for seed {seed} is ' +
                         f'{len(cards)}, which is not a multiple of 52.')
    num_decks = len(cards) // 52
    return shoe.Shoe(cards, num_decks=num_decks)


def _target_scores_from_percentiles(
        target_percentiles: typing.List,
        cards: typing.Union[str, typing.Iterable[card.Card]],
        payout: typing.Optional[payout_table.PayoutTable] = None
        ) -> typing.List:
    """Returns scores at given percentiles for shoe.

    Returns score that falls at the given percentiles in the score distribution
    for this shoe. Target percentiles can be chosen such that various agents
    will reach a target a given percent of the time.

    Median of min: .8781156156673329
    Median of med: .9761182362207991
    Median of max: .9960662058620562

    For example, targets of
    .8781156156673329, .9761182362207991, .9960662058620562

    simple:     - 36  * 35  ** 18  *** 10
    First-hand: - 17  * 28  ** 25  *** 30
    full-shoe:  - 14  * 26  ** 25  *** 35

    For current (non percentile) targets:

    simple:     - 0  * 62  ** 18  *** 20
    First-hand: - 0  * 25  ** 35  *** 40
    full-shoe:  - 0  * 20  ** 30  *** 50


    For two decks, the simple, first-hand and hand-stats agents achieve the
    following percentiles as follows:
    95  0.61505687, 0.72870219, 0.76139234
    90  0.69929298, 0.81515289, 0.84216896
    85  0.75556875, 0.86389035, 0.88641032
    80  0.79408326, 0.89646115, 0.9152023
    75  0.82332426, 0.91988554, 0.93659295
    70  0.85056081, 0.93893159, 0.95408972
    65  0.87187551, 0.95403066, 0.96657665
    60  0.89143657, 0.96748419, 0.97578543
    55  0.90776499, 0.976154  , 0.98275122
    50  0.92359389, 0.98293907, 0.98802613
    45  0.93795933, 0.98771156, 0.99153354
    40  0.95044243, 0.99142876, 0.99431522
    35  0.96206082, 0.99386352, 0.9960976
    30  0.97265264, 0.99598682, 0.99749645
    25  0.98101354, 0.99753112, 0.99845621
    20  0.98777126, 0.9986077 , 0.99913866
    15  0.99259128, 0.99930222, 0.99960177
    10  0.99630288, 0.99975377, 0.99985178
     5  0.99877473, 0.99994256, 0.99996889


    Args:
        target_percentiles: target percentiles
        cards: list of cards in shoe to evaluate
        payout: payout table to use, or default if not specified.

    Returns: List of target scores (integers) of same length as percentiles.
    """
    scores = list()
    dist, _, _ = calc_score_distribution(cards, payout)
    dist_scores = np.array([k for k in dist.keys()])
    dist_pct = np.array([v for v in dist.values()])
    for p in target_percentiles:
        if p >= 1.0:
            idx = -1
        else:
            idx = np.where(dist_pct > p)[0][0]
        scores.append(dist_scores[idx])
    return scores


def scores_to_json_str(scores: typing.List[ShoeTargetScores]) -> str:
    """Converts list of scores to JSON output.

    Returns: JSON string as '[{"<seed>": [<s1>, <s2>, <s3>, <s4>]}, ...]'
    """
    data = [{str(s[0]): s[1]} for s in scores]
    return json.dumps(data)


def calc_score_distribution(
        cards: typing.Union[str, typing.Iterable[card.Card]],
        payout: typing.Optional[payout_table.PayoutTable] = None,
        hs_range: float = 1.0, adv_range: float = 1.0):
    payout = payout or payout_table.PayoutTable.default()
    if isinstance(cards, str):
        cards = card.card_list(cards)
    else:
        cards = list(cards)
    shoe_size = len(cards)
    last_index = shoe_size - 10
    # payouts is a 2d array of multi-sets (counters), where payouts[p][h] is the
    # multiset of payouts for hands starting at position p and holding h cards.
    payouts = [[collections.Counter() for _ in range(6)]
               for _ in range(last_index + 1)]
    for p in range(last_index + 1):
        hand_at_p = hand.Hand(cards[p:p + 5])
        for held in hand_at_p.discard_combinations_iter():
            replacements = cards[p + 5:p + 5 + held.num_empty_slots()]
            filled = held.add(replacements)
            payouts[p][held.size()].update((int(filled.payout(payout)),))
    # Now compute score distributions for hands starting at position p, filling
    # from the bottom up.
    distributions = [collections.Counter() for _ in range(last_index + 1)]
    # Compute actual payout for each of the 32 possible ways to hold cards
    # at each hand position.
    for p in range(last_index, -1, -1):
        for h in range(6):
            next_p = p + 5 + (5 - h)
            for score, count in payouts[p][h].items():
                if next_p > last_index:
                    distributions[p].update({score: count})
                else:
                    for next_score, next_count in distributions[next_p].items():
                        distributions[p].update(
                            {score + next_score: count * next_count})
    # normalize and sort first distribution
    dist = distributions[0]
    total = 0
    for v in dist.values():
        total += v
    norm_dist = collections.OrderedDict()
    scores = sorted(dist.keys())
    cumulative = 0
    for s in scores:
        cumulative += dist[s]
        norm_dist.update({s: cumulative / total})
    return norm_dist, collections.OrderedDict(sorted(dist.items())), total


def _play_game_task(game: poker_game.PokerGame,
                    agents: typing.List[typing.Union[
                        player_agent.PlayerAgent,
                        typing.List[player_agent.PlayerAgent]]]) -> GameInfo:
    """Play game with one or more agents.

    Args:
        game: game to play. A copy will be made for each element of agents.
        agents: list of one or more agents or list of agents. If an element
            is a single agent then it will play the game; if an iterable then
            the first agent in the list will play but the others will analyze
            and add their data to the move's info field.

    Returns: (scores, num_hands, histories), each of which are lists of scores,
        number of hands played and histories for each agent.
    """
    scores_q = collections.deque()
    num_hands_q = collections.deque()
    history_q = collections.deque()
    for agent in agents:
        game_copy = game.copy()
        player_agent.play_game(game_copy, agent)
        scores_q.append(game_copy.score)
        num_hands_q.append(game_copy.hand_number)
        history_q.append(game_copy.history)
    return list(scores_q), list(num_hands_q), list(history_q)


def play_games(games: typing.Iterable[poker_game.PokerGame],
               agents: typing.Union[
                   player_agent.PlayerAgent,
                   typing.List[typing.Union[
                       player_agent.PlayerAgent,
                       typing.List[player_agent.PlayerAgent]]]],
               num_processes: typing.Optional[int] = None,
               seed: typing.Optional[np.random.SeedSequence] = None
               ) -> typing.List[GameInfo]:
    """Play list of games with one or more agents.

    Args:
        games: list of games to play.
        agents: a single agent, list agents, or list of lists of agents. If an
            element is a single agent then it will play the game; if an iterable
            then the first agent in the list will play but the others will
            analyze and add their data to the move's info field.
        num_processes: number of processes to use, or None for number of CPUs.
        seed: seed for rng used to shuffle shoes.

    Returns: list of (scores, num_hands, histories) tuples with scores, number
        of hands and move history for each game.
    """
    try:
        games_len = len(games)
    except TypeError:
        games_len = None
    agents = list(agents)
    task_func = functools.partial(_play_game_task, agents=agents)
    game_info_q = collections.deque()
    with tqdm.tqdm(total=games_len) as pbar:
        with multiprocessing.Pool(num_processes) as pool:
            for game_info in pool.imap(task_func, games):
                game_info_q.append(game_info)
                pbar.update()
    return list(game_info_q)


def default_target_agents(num_decks: int = 3, verbose: bool = False,
                          noise_iterations: int = 0,
                          seed: typing.Optional[np.random.SeedSequence] = None
                          ) -> typing.List[player_agent.PlayerAgent]:
    NOISE_VARIATIONS = 5
    NOISE = 0.75
    RISK = 0.5
    seed = seed or np.random.SeedSequence()
    seeds = seed.spawn(noise_iterations * NOISE_VARIATIONS)
    advantage = player_agent.advantage_table(num_decks)
    agents = [
        player_agent.SimplePlayerAgent(name='simple', verbose=verbose),
        player_agent.HandOnlyAgent(name='hands-only', verbose=verbose),
        player_agent.CardCountingAgent(name='card-counting', verbose=verbose),
        player_agent.CardCountingAgent(
            name='advantage', advantage=advantage, verbose=verbose),
        player_agent.CardCountingAgent(
            name='risk-averse', risk=-RISK, verbose=verbose),
        player_agent.CardCountingAgent(
            name='risk-seeking', risk=RISK, verbose=verbose)]

    for niter in range(1, noise_iterations + 1):
        agents += [
            player_agent.CardCountingAgent(
                name=f'hands-only-noise-{niter}', noise=NOISE,
                verbose=verbose, seed=seeds.pop()),
            player_agent.CardCountingAgent(
                name=f'card-counting-noise-{niter}', noise=NOISE,
                verbose=verbose, seed=seeds.pop()),
            player_agent.CardCountingAgent(
                name=f'risk-averse-noise-{niter}', risk=-RISK, noise=NOISE,
                verbose=verbose, seed=seeds.pop()),
            player_agent.CardCountingAgent(
                name=f'risk-seeking-noise-{niter}', risk=RISK, noise=NOISE,
                verbose=verbose, seed=seeds.pop()),
            player_agent.CardCountingAgent(
                name=f'advantage-noise-{niter}', advantage=advantage,
                noise=NOISE, verbose=verbose, seed=seeds.pop())]
    # Psychic
    agents.append(player_agent.PsychicAgent(name='psychic', verbose=verbose))
    return agents


def generate_games(num_games: int,
                   num_decks: int,
                   payout: typing.Optional[payout_table.PayoutTable] = None,
                   hand_limit: int = 0,
                   allow_short_shoe: bool = False,
                   seed: typing.Optional[np.random.SeedSequence] = None
                   ) -> typing.List[poker_game.PokerGame]:
    """Returns list of games with random shoes."""
    return [poker_game.PokerGame(num_decks=num_decks, payout=payout,
                                 hand_limit=hand_limit,
                                 allow_short_shoe=allow_short_shoe, seed=seed)
            for _ in range(num_games)]


def compute_targets(
        scores: np.ndarray, ranks: npt.ArrayLike = (2, 4, 6, 9)) -> np.ndarray:
    """Returns target scores for multiple games from array of agent scores.

    Args:
        scores: array of agent scores from games, of shape
            (num_games, num_agents).
        ranks: which ranks to select from scores in range [1, num_agents],
            where 1 is the lowest and num_agents is the highest score.

    Returns: targets, an ndarray of shape (num_games, num_ranks).
    """
    targets = np.sort(scores).take(np.array(ranks) - 1, axis=1)
    for col in range(len(ranks) - 2, -1, -1):
        targets[:, col] = np.where(targets[:, col] >= targets[:, col + 1],
                                   targets[:, col + 1] - 10, targets[:, col])
    assert not np.any(np.diff(targets, axis=1) <= 0)
    return targets


def target_scores(
        games: typing.Iterable[poker_game.PokerGame],
        ranks: npt.ArrayLike = (2, 4, 6, 9),
        agents: typing.Optional[
            typing.Iterable[player_agent.PlayerAgent]] = None,
        num_processes: typing.Optional[int] = None,
        ) -> typing.List[typing.List[int]]:
    """Returns target scores for given games.

    Args:
        games: list of games to evaluate.
        ranks: ranks from default agents to target.
        agents: list of agents to run, or None for default list.
        num_processes: number of processes to use, or None for number of CPUs.

    Returns: list of increasing targets for games.
    """
    if agents is None:
        agents = default_target_agents(num_decks=games[0].num_decks)
    game_info_list = play_games(games=games, agents=agents,
                                num_processes=num_processes)
    scores, _, _ = list(zip(*game_info_list))
    return compute_targets(scores, ranks)


def analyze_json_list_str(
        json_string: str,
        agents: typing.Optional[typing.List[
            typing.Union[player_agent.PlayerAgent,
                         typing.List[player_agent.PlayerAgent]]]] = None,
        agent_noise_iterations: int = 0,
        payout: typing.Optional[payout_table.PayoutTable] = None,
        hand_limit: int = 0,
        allow_short_shoe: bool = False,
        num_processes: typing.Optional[int] = None,
        seed: typing.Optional[np.random.SeedSequence] = None,
        ranks: typing.Optional[npt.ArrayLike] = None
        ) -> typing.List[ShoeTargetScores]:
    """Returns target scores for a json-encoded list of shoes.

    Args:
        json_string: JSON-encoded shoes to analyze, as string.
        agents: agents to use for evaluation, or None for default.
        agent_noise_iterations: number of noise iterations to use for agents,
            ignored if agents is not None.
        payout: payout to use in games, or None for default.
        hand_limit: hand-limit in games, or zero for unlimited.
        allow_short_shoe: whether to allow short-shoe at end of games.
        num_processes: number CPUs to use, or None for all available.
        seed: random seed to use in games to make noisy agents
            deterministic. Not to be confused with seeds passed in JSON string.
        ranks: which ranks to select from scores in range [1, num_agents],
            where 1 is the lowest and num_agents is the highest score. If
            None then all scores will be returned in original order.
    """
    seeds_q = collections.deque()
    games_q = collections.deque()
    shoe_data_list = json.loads(json_string)
    num_games = len(shoe_data_list)
    if seed is not None:
        rng_seeds = seed.spawn(num_games)
    else:
        rng_seeds = [None] * num_games
    for shoe_data, rng_seed in zip(shoe_data_list, rng_seeds):
        draw_shoe = _json_to_shoe(shoe_data)
        seeds_q.append(shoe_data['seed'])
        game = poker_game.PokerGame(
            num_decks=draw_shoe.num_decks, draw_shoe=draw_shoe, payout=payout,
            hand_limit=hand_limit, allow_short_shoe=allow_short_shoe,
            seed=rng_seed)
        games_q.append(game)
    if agents is None and games_q:
        agents = default_target_agents(num_decks=games_q[0].num_decks,
                                       noise_iterations=agent_noise_iterations)
        if ranks is not None and 10 not in ranks:
            agents = agents[:-1]  # don't bother with psychic
    game_info_list = play_games(games_q, agents, num_processes=num_processes)
    scores = [game_info[0] for game_info in game_info_list]
    if ranks is not None:
        scores = compute_targets(np.array(scores), ranks=ranks).tolist()
    return [target for target in zip(seeds_q, scores)]


def _probs_at_point(point: int, num_decks: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Returns probabilities of winning hands at given point.

    Args:
        point: point in deck, with zero being start.
        num_decks: number of decks iin full shoe.
        rng: random number generator to use for shuffling.

    Returns: array of shape (32, 12), with probabilities in [0, 1]
        for each of the 12 possible hands from royal flush to no hand,
        for each of the 32 possible ways to discard.
    """
    cards = shoe.shuffled_cards(num_decks=num_decks, rng=rng)
    if point + 5 == 0:
        delt = hand.Hand(cards[point:])
    else:
        delt = hand.Hand(cards[point:point+5])
    hs = hand_stats.HandStats(cards[point+5:])
    probs = [hs.prob_winning_hand(held)
             for held in delt.discard_combinations_iter()]
    if len(probs) < 32:
        print(f'delt: {delt}, point: {point}, len(cards): {len(cards)}, '
              f'len(probs): {len(probs)}')
    return np.array(probs)  #.max(axis=0)


def probabilities_at_points(
        seed: typing.Optional[np.random.SeedSequence] = None,
        points: typing.Optional[typing.List] = None,
        num_decks: int = 3,
        ) -> np.ndarray:
    """Returns probabilities of winning hands at given points in shoe.

    Args:
        seed: random seed to use.
        points: list of points to test, or None for all in shoe.
        num_decks: number of decks in a full shoe.

    Returns: probabilities, an array of shape (len(points), 32, 12).
    """
    if points is None:
        points = np.arange(52 * num_decks - 5)
    rng = np.random.default_rng(seed)
    probs = [_probs_at_point(point, num_decks, rng)
             for point in points]
    return np.array(probs)


def probabilities_at_points_trials(
        num_trials: int,
        points: typing.List[int],
        num_decks: int,
        num_processes: typing.Optional[int] = None,
        seed: typing.Optional[np.random.SeedSequence] = None):
    """Returns list of probability reports at given points from multiple trials.

    Args:
        num_trials: number of trials to run.
        points: list of points to check.
        num_decks: number decks in the shoe.
        num_processes: number of CPUs to use, or None for all.
        seed: random seed to use.

    Returns: probabilities, an array of shape (num_trials, len(points), 32, 12).
    """
    seeds = np.random.SeedSequence(seed).spawn(num_trials)
    task_func = functools.partial(
        probabilities_at_points, points=points, num_decks=num_decks)
    probs = collections.deque()
    with tqdm.tqdm(total=num_trials) as pbar:
        with multiprocessing.Pool(num_processes) as pool:
            for data in pool.imap_unordered(task_func, seeds):
                probs.append(data)
                pbar.update()
    return np.array(probs)


if __name__ == '__main__':
    # Compute probabilities of winning hands with replacement
    import pandas as pd
    import numpy as np

    _num_trials = 10000
    _num_decks = 3

    _shoe_size = _num_decks * 52
    _points = np.arange(-50, -8) + _shoe_size
    _probs = probabilities_at_points_trials(
        num_trials=_num_trials, points=_points, num_decks=_num_decks)

    freq_names = list(hand_stats.HandFrequency._fields)
    probs_df = pd.DataFrame(_probs.mean(axis=0), columns=freq_names,
                            index=_points)
    with pd.option_context('display.precision', 3):
        print(probs_df)
