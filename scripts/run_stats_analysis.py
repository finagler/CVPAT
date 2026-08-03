import copy
import itertools

import numpy as np
import pandas as pd
from scipy import stats
import timeit
import typing

from stats import hand_stats
from game import shoe
from game import hand
from game import poker_game
from game import payout_table
from agent import player_agent

np.set_printoptions(suppress=True, precision=4)
np.seterr(all='raise')


def ci(
        data: np.ndarray, confidence: float=0.95) -> float:
    """Returns estimated mean and confidence interval around it.

    Args:
        data: values for which mean is to be estimated.
        confidence: probability that reported mean is within returned interval.

    Returns:
        interval: mean of data = mean ± interval, with given confidence.
    """
    mean = data.mean()
    try:
        interval = stats.norm.interval(
            confidence, loc=mean, scale=stats.sem(data))
    except FloatingPointError:
        return np.nan
    return interval[1] - mean


def compute_mean_expected_payout_with_discard(
        shoe_size: int = 208,
        num_trials: int = 100,
        num_decks: int = 4):
    # Test: loop and test random hands
    payouts: typing.List[float] = []
    for trial in range(num_trials):
        draw_shoe = shoe.Shoe.shoe_with_num_cards_remaining(
            shoe_size, num_decks=num_decks)
        delt = draw_shoe.draw_cards()
        hs = hand_stats.HandStats(draw_shoe)
        best_payout = hs.best_expected_payout(hand.Hand(delt))
        payouts.append(best_payout.value)
    payouts_array = np.array(payouts)
    mean = payouts_array.mean()
    conf = ci(payouts_array)
    print(f'{shoe_size} cards: mean: {mean:.3f} ± {conf:.3f}')


def distribution_of_mean_expected_payout(
        shoe_size: int = 104, num_shoes: int = 11,
        num_hands: int = 100, num_decks: int = 4) -> np.ndarray:
    payouts = np.full((num_shoes, num_hands), np.nan)
    print(f'shoe_size: {shoe_size}')
    for shoe_num in range(num_shoes):
        draw_shoe = shoe.Shoe.shoe_with_num_cards_remaining(
            shoe_size, num_decks=num_decks)
        for hand_num in range(num_hands):
            delt = draw_shoe.draw_cards()
            hs = hand_stats.HandStats(draw_shoe)
            best_payout = hs.best_expected_payout(hand.Hand(delt))
            payouts[shoe_num, hand_num] = best_payout.payout
            draw_shoe.add_cards(delt)
        mean = payouts[shoe_num].mean()
        conf = ci(payouts[shoe_num])
        print(f'    mean: {mean:.2f} ± {conf:.2f}')
    print()
    shoe_means = payouts.mean(axis=1)
    payouts = np.sort(payouts[shoe_means.argsort()])  # Sort by hand mean and row mean
    shoe_std = shoe_means.std()
    for row in range(num_shoes):
        mean = payouts[row].mean()
        conf = ci(payouts[row])
        print(f'    mean: {mean:.2f} ± {conf:.2f}')
    print(f'    overall mean: {payouts.mean():.2f} ± {ci(payouts.flatten()):.3f}, shoe std: {shoe_std}')
    return payouts


def distribution_of_mean_expected_payout_for_shoe(
        draw_shoe: shoe.Shoe, num_hands: int = 100,
        payout_table: typing.Optional[payout_table.PayoutTable] = None,
        out=None) -> np.ndarray:
    print(f'draw_shoe.size: {draw_shoe.size()}')
    print(f'draw_shoe: {draw_shoe.cards}')
    if out is None:
        out = np.full((num_hands), np.nan)
    for hand_num in range(num_hands):
        delt = draw_shoe.draw_cards()
        hs = hand_stats.HandStats(draw_shoe, payout=payout_table)
        best_payout = hs.best_expected_payout(hand.Hand(delt))
        out[hand_num] = best_payout.payout
        draw_shoe.add_cards(delt)
    print(f'    mean: {out.mean():.2f} ± {ci(out):.2f}')
    return out


def compute_expected_payout_distribution(
        start_size: int = 208,
        step_size: int = -10,
        trials_per_size: int = 100,
        num_decks: int = 4,
        payout_table: typing.Optional[payout_table.PayoutTable] = None):
    shoe_sizes = np.arange(start_size, 9, step_size)
    payouts = np.full((len(shoe_sizes), trials_per_size), np.nan)
    for i, shoe_size in enumerate(shoe_sizes):
        for trial in range(trials_per_size):
            draw_shoe = shoe.Shoe.shoe_with_num_cards_remaining(
                shoe_size, num_decks=num_decks)
            delt = draw_shoe.draw_cards()
            hs = hand_stats.HandStats(draw_shoe, payout=payout_table)
            best_payout = hs.best_expected_payout(hand.Hand(delt))
            payouts[i, trial] = best_payout.payout
        mean = payouts[i].mean()
        conf = ci(payouts[i])
        qtiles = np.quantile(payouts[i], np.linspace(0.0, 1.0, num=11))
        print(f'{shoe_size} cards: mean: {mean:.3f} ± {conf:.3f}, qtiles={qtiles}')
    return np.sort(payouts)


def play_games(num_games: int = 10, num_decks: int = 4,
               hands_per_game: int = 100,
               discard_value: float = 0.0,
               compare_kkind_and_straights: bool = False,
               print_hand_prob_table: bool = False):
    game = poker_game.PokerGame(num_decks=num_decks)
    winnings = []
    expected_payout = []
    total_expected_prob = np.zeros(10)
    total_winning_freq = np.zeros(10)
    num_held = []
    num_hands = []
    all_expected_payouts = []
    start_shoe_size = num_decks * 52
    kind_vs_straight_data = []
    for trial in range(num_games):
        done = False
        while not done and game.hand_number <= hands_per_game:
            hs = hand_stats.HandStats(game.draw_shoe, game.payout)
            expected = hs.best_expected_payout_per_num_held(game.hand)
            payout_by_held = [p.value for p in expected]
            all_expected_payouts.append(payout_by_held)
            frac_shoe_remaining = game.draw_shoe.size() / start_shoe_size
            payout_by_held += discard_value * frac_shoe_remaining * np.arange(6)
            best_hold = np.argmax(payout_by_held)
            best = expected[best_hold]

            if compare_kkind_and_straights:
                straight_table = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0]) * game.payout.table
                kkind_table = np.array(   [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]) * game.payout.table
                straight_only_payout = payout_table.PayoutTable(*straight_table)
                kkind_only_payout = payout_table.PayoutTable(*kkind_table)
                hs_straight = hand_stats.HandStats(game.draw_shoe, straight_only_payout)
                hs_kkind = hand_stats.HandStats(game.draw_shoe, kkind_only_payout)
                expected_straight = hs_straight.best_expected_payout_per_num_held(game.hand)
                best_straight_hold = np.argmax([p.value for p in expected_straight])
                best_straight = expected_straight[best_straight_hold]
                expected_kkind = hs_kkind.best_expected_payout_per_num_held(game.hand)
                best_kkind_hold = np.argmax([p.value for p in expected_kkind])
                best_kkind = expected_kkind[best_kkind_hold]

                if (best.held == best_straight.held and best.held == best_kkind.held):
                    choice = 'both'
                elif (best.held == best_straight.held and best.held != best_kkind.held):
                    choice = 'straight'
                elif (best.held != best_straight.held and best.held == best_kkind.held):
                    choice = 'kkind'
                else:
                    choice = 'neither'

                prob_df = probability_table_for_hand(
                    draw_shoe=game.draw_shoe, delt=game.hand)
                prob_df_string = (100 * prob_df).to_string(float_format='%.4f')
                kind_vs_straight_data.append((
                    game.hand, game.draw_shoe.size(), choice,
                    copy.deepcopy(best), copy.deepcopy(best_straight),
                    copy.deepcopy(best_kkind), prob_df_string))

            num_held.append(best.held.size())
            expected_payout.append(best.value)
            total_expected_prob += hs.prob_winning_hand(best.held)
            winnings.append(game.discard_and_replace(best.held))
            total_winning_freq += game.hand.eval_hand()
            done = game.draw_new_hand()
        num_hands.append(min(game.hand_number, hands_per_game))
        game.reset()
    winnings_a = np.array(winnings)
    expected_winnings_a = np.array(expected_payout)
    num_held_a = np.array(num_held)
    num_hands_a = np.array(num_hands)
    all_expected_payouts_a = np.array(all_expected_payouts)
    print(f'games played: {num_games}')
    print(f'hands played: {num_hands_a.sum()}')
    print(f'mean hands / game: {num_hands_a.mean():0.3} ± {ci(num_hands_a):0.3}')
    print(f'mean cards held / hand: {num_held_a.mean():0.3} ± {ci(num_held_a):0.3}')
    print(f'mean expected payout by num cards held: {all_expected_payouts_a.mean(axis=0)} ± {ci(all_expected_payouts_a)}')
    print(f'std expected payout by num cards held: {all_expected_payouts_a.std(axis=0)}')
    print(f'skew expected payout by num cards held: {stats.skew(all_expected_payouts_a, axis=0)}')
    print(f'mean expected payout / hand: {expected_winnings_a.mean():0.3} ± {ci(expected_winnings_a):0.3}')
    print(f'mean actual payout / hand: {winnings_a.mean():0.3} ± {ci(winnings_a):0.3}')
    print(f'max/std expected payout / hand: {expected_winnings_a.max(initial=0):0.3}, {expected_winnings_a.std():0.3}')
    print(f'expected total win frequency by hand type: {total_expected_prob}')
    print(f'actual total win frequency by hand type: {total_winning_freq}')
    print(f'difference: {total_winning_freq - total_expected_prob}', flush=True)

    if compare_kkind_and_straights:
        print('\nProbability analysis for each hand')
        for (full_hand, shoe_size, choice, best, best_straight, best_kkind,
             prob_df_string) in kind_vs_straight_data:
            print(f'\n shoe size: {shoe_size}, hand: {full_hand}, holding: {best.held}, goal: {choice}')
            print(f'best expected payout: full: {best.value:.3} {best.held}, straight-only: {best_straight.value:.3} {best_straight.held}, kkind-only: {best_kkind.value:.3} {best_kkind.held}')
            print(prob_df_string)


def time_analysis(shoe_size: int = 208, num_decks: int = 4, num_trials=1000):
    # Test: loop and test random hands
    setup = f"""
from game import hand_stats
from game import shoe
import numpy as np

draw_shoe = shoe.Shoe.shoe_with_num_cards_remaining(
    {shoe_size}, num_decks={num_decks})
delt = draw_shoe.draw_cards()
# hs = hand_stats.HandStats(draw_shoe)
"""
    # execute = 'hs.best_expected_payout(delt)'
    execute = 'hs = hand_stats.HandStats(draw_shoe.cards)'
    return timeit.timeit(execute, setup=setup, number=num_trials)


def sample_winnings_vs_expected(
        num_hands: int = 10, num_held: int = 5, num_decks: int = 4,
        size_shoe: int = 208):
    game = poker_game.PokerGame(num_decks=num_decks)
    for trial in range(num_hands):
        if game.draw_shoe.size() > size_shoe:
            _ = game.draw_shoe.draw_cards(game.draw_shoe.size() - size_shoe)
        hs = hand_stats.HandStats(game.draw_shoe, game.payout)
        best = hs.best_expected_payout_per_num_held(game.hand)[num_held]
        held = best.held
        expected_freq = hs.freq_winning_hands(held)
        hand_eval = np.zeros(expected_freq.shape)
        for cards in itertools.combinations(game.draw_shoe.cards, 5 - num_held):
            test_hand = hand.Hand(held.cards + list(cards))
            hand_eval += test_hand.eval_hand()
        if np.any(expected_freq != hand_eval):
            print(f'expected: {expected_freq}, hand_eval: {hand_eval}')
            print(f'hand: {game.hand}, held: {held}')
            print(f'shoe: {game.draw_shoe.cards}')
            break



def probability_table_for_hand(
        draw_shoe: shoe.Shoe, delt: hand.Hand) -> pd.DataFrame:
        """Returns DataFrame of probabilities for each hand type x cards held.

        Args:
            draw_shoe: shoe drawn from.
            delt: cards currently in hand. This will normally be a full five
                card hand, though it's not checked.

        Returns:
            DataFrame of probabilities with shape (10, 32)
        """
        hs = hand_stats.HandStats(draw_shoe)
        held_list = [h for h in delt.discard_combinations_iter()]
        probabilities = np.array([hs.prob_winning_hand(h) for h in held_list]).T
        held_labels = [''.join([c.__repr__() for c in h.cards]) for h in held_list]
        return pd.DataFrame(probabilities, index=payout_table.HANDS_TYPE_NAMES,
                            columns=held_labels)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    game = poker_game.PokerGame(hand_limit=20)
    agent = player_agent.PsychicAgent(game, verbose=True)
    agent.play_game()
    print(game.history)
