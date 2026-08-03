import copy
import pathlib
import typing

import numpy as np

from game import hand
from game import poker_game
from stats import hand_stats
from game import payout_table
from agent import psychic_solver
from game import shoe


def advantage_table(num_decks: int) -> np.ndarray:
    data_dir = pathlib.Path(__file__).parent.parent / 'data'
    if num_decks == 2:
        return np.load(str(data_dir / 'advantage-table-2deck.npy'))
    if num_decks == 3:
        return np.load(str(data_dir / 'advantage-table-3deck.npy'))
    if num_decks == 4:
        return np.load(str(data_dir / 'advantage-table-4deck.npy'))
    raise FileNotFoundError(
        f'Cannot find advantage file for {num_decks} decks')


class PlayerAgent(object):
    def __init__(self, verbose: bool = False, name: str = '',
                 seed: typing.Optional[np.random.SeedSequence] = None):
        self.verbose = verbose
        self.name = name
        self.seed = seed or np.random.SeedSequence()

    def get_move(self, game: poker_game.PokerGame
                 ) -> typing.Tuple[hand.Hand, typing.Dict]:
        """Select a move to make. Does not modify game.

        Args:
            game: current game state.

        Returns: (held, info), where held is the cards to hold and info is
            a dictionary of other information that might be useful for stats
            or debugging.
        """
        raise NotImplementedError()

    def play(self, game: poker_game.PokerGame,
             verbose: bool = False
             ) -> typing.List[poker_game.GameMove]:
        """Play game using this agent, modifying self.game.

        Returns: trajectory from game.
        """
        return play_game(game, self, verbose)


class InteractivePlayerAgent(PlayerAgent):
    def get_move(self, game: poker_game.PokerGame
                 ) -> typing.Tuple[hand.Hand, typing.Dict]:
        """Select a move to make. Does not modify game.

        Args:
            game: current game state.

        Returns: (held, info), where held is the cards to hold and info is
            a dictionary of other information that might be useful for stats
            or debugging.
        """
        held_str = input(f'    hold from {game.hand}: ')
        if held_str.lower() == 'quit':
            game.game_over = True
            return game.hand, dict()
        return hand.Hand(held_str), dict()


class CardCountingAgent(PlayerAgent):
    """Agent that uses contents (but not order) of shoe to find best move."""
    def __init__(self,
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 verbose: bool = False, name: str = '',
                 advantage: typing.Union[None, typing.List[int],
                                         np.ndarray] = None,
                 risk: float = 0.0, noise: float = 0.0,
                 seed: typing.Optional[np.random.SeedSequence] = None):
        super().__init__(verbose, name, seed)
        self.payout = payout
        if advantage is not None:
            self.advantage = np.array(advantage)
        else:
            self.advantage = np.zeros((6,))
        self.risk = risk
        self.noise = noise

    def _advantage_from_lookup(self, game: poker_game.PokerGame) -> np.array:
        if len(self.advantage.shape) == 1:
            return self.advantage
        # Shoe position is position of start of current hand in the shoe,
        # starting at zero for the first hand. This is equivalent to counting
        # shoe_size cards from the end of the shoe. If the advantage table
        # is too small (e.g. if we're playing with more decks than the table
        # was built for) then we just use the first row until we get into the
        # table. It doesn't actually change much until about 50 cards before
        # the end.
        shoe_position = max(-game.draw_shoe.size(),
                            -self.advantage.shape[1] + 5)
        adv = self.advantage[game.hands_remaining, shoe_position - 5]
        return adv

    def _hand_stats(self, game: poker_game.PokerGame) -> hand_stats.HandStats:
        seed = np.random.SeedSequence(
            [self.seed.entropy, self.seed.spawn_key,
             game.seed.entropy, game.seed.spawn_key])
        return hand_stats.HandStats(game.draw_shoe.cards, self.payout,
                                    rng=np.random.default_rng(seed))

    def get_move(self, game: poker_game.PokerGame
                 ) -> typing.Tuple[hand.Hand, typing.Dict]:
        hs = self._hand_stats(game)
        advantage = self._advantage_from_lookup(game)
        hv = hs.best_expected_payout(game.hand, advantage=advantage,
                                     risk=self.risk, noise=self.noise)
        info = dict()
        if self.verbose:
            if self.name:
                info['name'] = self.name
            info['held'] = hv.held
            info['value'] = hv.value
            best = hs.best_expected_payout_per_num_held(game.hand)
            hand_value = np.array([b.value for b in best])
            info['hand_value'] = hand_value
            info['advantage'] = advantage
            info['total_value'] = hand_value + advantage
        return hv.held, info


def _jack_plus_cards(delt: hand.Hand):
    """Returns deck with only jack through ace held."""
    cards = [c for c in delt.cards
             if c.rank > 10 or c.rank == 1]
    return hand.Hand(cards)


class SimplePlayerAgent(PlayerAgent):
    def __init__(self,
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 verbose: bool = False, name: str = ''):
        super().__init__(verbose, name)
        self.payout = payout or payout_table.PayoutTable.default()

    def get_move(self, game: poker_game.PokerGame
                 ) -> typing.Tuple[hand.Hand, typing.Dict]:
        current_payout = game.hand.payout(self.payout)
        held = copy.deepcopy(game.hand)
        if current_payout:
            # discard as many cards as possible without lowering current payout
            for h in game.hand.discard_combinations_iter():
                if (h.size() < held.size() and
                        h.payout(self.payout) >= current_payout):
                    held = h
        else:
            held = _jack_plus_cards(held)
        info = dict()
        if self.verbose:
            info['name'] = self.name
            info['current_payout'] = current_payout
        return held, info


class HandOnlyAgent(CardCountingAgent):
    """Agent that uses hand and number of decks to find best move.

    Agent that knows hand and game configuration but does not have access to
    the contents of the shoe. This is equivalent to the CardCountingAgent where
    every hand is treated as if it had a full shoe.
    """
    def __init__(self,
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 verbose: bool = False, name: str = '',
                 advantage: typing.Union[None, typing.List[int],
                                         np.ndarray] = None,
                 risk: float = 0.0, noise: float = 0.0):
        super().__init__(payout=payout, verbose=verbose, name=name,
                         advantage=advantage, risk=risk, noise=noise)

    def _hand_stats(self, game: poker_game.PokerGame) -> hand_stats.HandStats:
        full_shoe = shoe.Shoe.shoe_excluding_cards(
            game.hand.cards, num_decks=game.num_decks)
        return hand_stats.HandStats(full_shoe.cards, self.payout)


class PsychicAgent(PlayerAgent):
    """Agent that cheats by looking at all cards.

    PsychicAgent will always get the absolute best possible score on any shoe,
    and should be treated as an upper bound rather than what a player could
    get on their first time playing a shoe.

    Note: PsychicAgent computes the solution table for a game on the first
    call to get_move() and then references it for subsequent calls. The agent
    keeps track of the hash of the current game being played and will regenerate
    the solution if that changes, but any changes made to the game object
    itself (like the payout table) will not be noticed until reset_game is
    called explicitly.
    """
    def __init__(self, verbose: bool = False, name: str = ''):
        # PsychicSolver computes the whole value table for all offsets and
        # hand limits on the first move, and then references it for all
        # subsequent calls to get_move.
        super().__init__(verbose, name)
        self._game_hash = None  # hash of current game being played
        self.values = self.num_held = self.held = None

    def reset_game(self, game: poker_game.PokerGame):
        self._game_hash = hash(game)
        cards = game.hand.cards + game.draw_shoe.cards
        solver = psychic_solver.PsychicSolver(
            cards=cards, hand_limit=game.hand_limit, payout=game.payout,
            allow_short_shoe=game.allow_short_shoe)
        self.values, self.num_held, self.held = solver.solve_all()

    def _is_continuation(self, game: poker_game.PokerGame):
        return hash(game) == self._game_hash

    def get_move(self, game: poker_game.PokerGame
                 ) -> typing.Tuple[hand.Hand, typing.Dict]:
        if not self._is_continuation(game):
            self.reset_game(game)
        offset = -game.draw_shoe.size() - 5
        if game.hand_limit:
            hands_remaining = game.hands_remaining
            cum_value = self.values[hands_remaining, offset]
            num_held = self.num_held[hands_remaining, offset]
            next_offset = offset + 5 + (5 - num_held)
            if hands_remaining - 1 >= 0 and next_offset < self.values.shape[-1]:
                value = (cum_value -
                         self.values[hands_remaining - 1, next_offset])
            else:
                value = cum_value
        else:
            cum_value = self.values[offset]
            num_held = self.num_held[offset]
            next_offset = offset + 5 + (5 - num_held)
            if next_offset < self.values.shape[-1]:
                value = cum_value - self.values[next_offset]
            else:
                value = cum_value
        held = self.held[offset][num_held]
        info = dict()
        if self.verbose:
            if self.name:
                info['name'] = self.name
            info['held'] = held
            info['cum_value'] = cum_value
            info['value'] = value
        return held, info


def play_game(game: poker_game.PokerGame,
              agent: typing.Union[PlayerAgent, typing.List[PlayerAgent]],
              verbose: bool = False
              ) -> typing.List[poker_game.GameMove]:
    """Play one full game using agent.

    Args:
        game: game engine to play
        agent: agent or list of agents to pick which moves to make. If a list
            then the first agent in the list makes the moves but the others
            are also run and add their analysis to the info field in the
            trajectory.
        verbose: if True then print which cards are held to console.

    Returns:
        trajectory: history of the game.
    """
    try:
        agents = list(agent)
    except TypeError:
        agents = list([agent])

    done = game.draw_new_hand()
    while not done:
        helds, infos = list(zip(*[agent.get_move(game) for agent in agents]))
        if verbose and infos:
            with np.printoptions(suppress=True, precision=1,
                                 formatter={'float': '{:>4.1f}'.format}):
                print(f'    held:  {infos[0]["held"]}')
                print(f'    value: {infos[0]["value"]:0.1f}')
                print(f'    hand:  {infos[0]["hand_value"]}')
                print(f'    adv:   {infos[0]["advantage"]}')
                print(f'    total: {infos[0]["total_value"]}')
        reward = game.discard_and_replace(helds[0], infos)
        if verbose and reward:
            print(f'    payout: {reward}, total: {game.score}')
        done = game.draw_new_hand()
    return game.history
