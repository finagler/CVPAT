from stats import hand_stats
from game import payout_table
from game import card
from game import shoe
from game import hand

import pytest
import numpy as np
from numpy import testing as npt
from scipy import special


def comb(n, k):
    return special.binom(n, k).astype(int)


def test_size():
    cards = (card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.CLUB))
    draw_shoe = shoe.Shoe.shoe_excluding_cards(cards)
    test_hs = hand_stats.HandStats(draw_shoe.cards)
    assert test_hs.size() == 50


# Expected winnings
def test_freq_winning():
    test_hs = hand_stats.HandStats(shoe.Shoe().cards)
    winning = test_hs.freq_winning_hands(held=hand.Hand())
    # Expected frequency for one deck, as calculated at
    # https://en.wikipedia.org/wiki/Poker_probability
    expected = np.array([4, 36, 0, 624, 3744, 5108,
                         10200, 54912, 123552, 337920, 760320,
                         1302540])
    np.testing.assert_equal(winning, expected)


def test_freq_winning_2decks():
    test_hs = hand_stats.HandStats(shoe.Shoe(num_decks=2).cards)
    winning = test_hs.freq_winning_hands(held=hand.Hand())
    expected = [4 * 2 ** 5,                                  # royal flush
                4 * 10 * 2 ** 5 - (4 * 2 ** 5),              # straight flush
                13 * comb(8, 5),                             # 5 kind
                13 * comb(8, 4) * 12 * comb(8, 1),           # 4 kind
                13 * comb(8, 3) * 12 * comb(8, 2),           # full house
                4 * comb(26, 5) - (4 * 10 * 2 ** 5),         # flush
                10 * 8 ** 5 - (4 * 10 * 2 ** 5),             # straight
                13 * comb(8, 3) * comb(12, 2) * 8 ** 2,      # 3 kind
                (comb(13, 2) * comb(8, 2) ** 2 *
                 11 * comb(8, 1) -
                 (4 * comb(13, 2) * 11 * 2)),                # 2 pair
                (4 * comb(8, 2) * comb(12, 3) * 8 ** 3 -
                 (4 * 4 * comb(12, 3) * 2 ** 3)),            # jack-high pair
                (9 * comb(8, 2) * comb(12, 3) * 8 ** 3 -
                 (4 * 9 * comb(12, 3) * 2 ** 3)),            # low pair
                (comb(13, 5) - 10) * (8 ** 5 - 4 * 2 ** 5)]  # no hand
    np.testing.assert_equal(winning, expected)


def test_freq_all_draws():
    cards = (card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.DIAMOND))
    draw_shoe = shoe.Shoe.shoe_excluding_cards(cards, num_decks=2)
    test_hs = hand_stats.HandStats(draw_shoe.cards)
    # Ways to draw 3 cards from 102 cards: (102 choose 3) = 171700
    assert test_hs.freq_all_draws(held=hand.Hand(cards)) == 171700


def test_prob_winning_hand():
    test_hs = hand_stats.HandStats(shoe.Shoe().cards)
    # Expected probabilities for one deck, as calculated at
    # https://en.wikipedia.org/wiki/Poker_probability
    expected_pct = [0.000154,  # Royal flush
                    0.00139,   # Straight flush
                    0.0,       # Five-of-a-kind
                    0.02401,   # Four-of-a-kind
                    0.1441,    # Full house
                    0.1965,    # Flush
                    0.3925,    # Straight
                    2.1128,    # Three-of-a-kind
                    4.7539,    # Two pair
                    13.00212,  # Jack high pair
                    29.25478,  # Low pair
                    50.1177]   # No hand
    np.testing.assert_allclose(
        100 * test_hs.prob_winning_hand(held=hand.Hand()),
        expected_pct, atol=1e-4)
    np.testing.assert_allclose(np.sum(expected_pct), 100.0, atol=1e-4)


def test_expected_payout():
    payout = payout_table.PayoutTable(
        10000, 1000, 200, 80, 70, 60, 50, 20, 10, 5, 1)
    test_hs = hand_stats.HandStats(shoe.Shoe().cards, payout=payout)
    expected_payout = 0.01 * (0.000154 * 10000 +  # Royal flush
                              0.00139 * 1000 +    # Straight flush
                              0 * 200 +           # Five-of-a-kind
                              0.0241 * 80 +       # Four-of-a-kind
                              0.1441 * 70 +       # Full house
                              0.1965 * 60 +       # Flush
                              0.3925 * 50 +       # Straight
                              2.1128 * 20 +       # Three-of-a-kind
                              4.7539 * 10 +       # Two pair
                              13.00212 * 5 +      # Jack high pair
                              29.25478 * 1)       # Low pair
    got_payout = test_hs.expected_payout(held=hand.Hand())
    assert got_payout == pytest.approx(expected_payout, abs=1e-2)


# Test hand_freq
def _run_hand_freq(hand_str: str, num_decks: int) -> hand_stats.HandFrequency:
    cards = card.card_list(hand_str)
    draw_shoe = shoe.Shoe.shoe_excluding_cards(cards, num_decks=num_decks)
    test_hs = hand_stats.HandStats(draw_shoe.cards)
    held = hand.Hand(cards)
    return test_hs.hand_freq(held=held)


def test_hand_freq_4deck_empty():
    freq = _run_hand_freq('', num_decks=4)
    assert freq.royal_flush == 4 * comb(4, 1)**5  # AKQJ10 same suit
    assert freq.straight_flush == 9 * 4 * 4**5  # X0-X1-X2-X3-X4, same suit
    assert freq.five_kind == 13 * comb(16, 5)  # XXXXX
    assert freq.four_kind == 13 * comb(16, 4) * 12 * 16  # XXXXY
    assert freq.full_house == 13 * comb(16, 3) * 12 * comb(16, 2)  # XXXYY

    hand_xxxyz = 13 * comb(16, 3) * comb(12, 2) * 16**2
    hand_xxxyz_same = 4 * 13 * comb(4, 3) * comb(12, 2) * 4**2
    assert freq.three_kind == hand_xxxyz - hand_xxxyz_same

    hand_xxyyz = comb(13, 2) * comb(16, 2)**2 * 11 * 16
    hand_xxyyz_same = 4 * comb(13, 2) * comb(4, 2)**2 * 11 * 4
    assert freq.two_pair == hand_xxyyz - hand_xxyyz_same

    hand_hhxyz = 4 * comb(16, 2) * comb(12, 3) * 16**3
    hand_hhxyz_flush = 4 * 4 * comb(4, 2) * comb(12, 3) * 4**3
    assert freq.jack_high_pair == hand_hhxyz - hand_hhxyz_flush


def test_hand_freq_2deck_empty():
    freq = _run_hand_freq('', num_decks=2)
    assert freq.royal_flush == 4 * 2**5  # AKQJ10, same suit
    assert freq.straight_flush == 4 * 9 * 2**5  # X0-X1-X2-X3-X4, same suit
    assert freq.five_kind == 13 * comb(8, 5)  # XXXXX
    assert freq.four_kind == 13 * comb(8, 4) * 12 * 8  # XXXXY
    assert freq.full_house == 13 * comb(8, 3) * 12 * comb(8, 2)  # XXXYY
    assert freq.three_kind == 13 * comb(8, 3) * comb(12, 2) * 8**2  # XXXYZ

    hand_xxyyz = comb(13, 2) * comb(8, 2)**2 * 11 * 8
    hand_xxyyz_flush = 4 * comb(13, 2) * 11 * 2
    assert freq.two_pair == hand_xxyyz - hand_xxyyz_flush

    hand_hhxyz = 4 * comb(8, 2) * comb(12, 3) * 8**3
    hand_hhxyz_flush = 4 * 4 * comb(12, 3) * 2**3
    assert freq.jack_high_pair == hand_hhxyz - hand_hhxyz_flush


def test_hand_freq_4deck_1ace():
    freq = _run_hand_freq('AC', num_decks=4)
    assert freq.royal_flush == 4**4  # A+KQJ10 clubs
    assert freq.straight_flush == 4**4  # A+2345 clubs
    assert freq.five_kind == comb(15, 4)  # A+AAAA

    hand_a_aaax = comb(15, 3) * 12 * 16
    hand_a_xxxx = 12 * comb(16, 4)
    assert freq.four_kind == hand_a_aaax + hand_a_xxxx

    hand_a_aaxx = comb(15, 2) * 12 * comb(16, 2)
    hand_a_axxx = 15 * 12 * comb(16, 3)
    assert freq.full_house == hand_a_aaxx + hand_a_axxx

    hand_a_aaxy = comb(15, 2) * comb(12, 2) * 16**2
    hand_a_xxxy = 12 * comb(16, 3) * 11 * 16
    hand_a_aaxy_flush = comb(3, 2) * comb(12, 2) * 4**2
    hand_a_xxxy_flush = 12 * comb(4, 3) * 11 * 4
    assert freq.three_kind == (hand_a_aaxy + hand_a_xxxy -
                               hand_a_aaxy_flush - hand_a_xxxy_flush)

    hand_a_axxy = 15 * 12 * comb(16, 2) * 11 * 16
    hand_a_xxyy = comb(12, 2) * comb(16, 2)**2
    hand_a_axxy_flush = 3 * 12 * comb(4, 2) * 11 * 4
    hand_a_xxyy_flush = comb(12, 2) * comb(4, 2)**2
    assert freq.two_pair == (hand_a_axxy + hand_a_xxyy -
                             hand_a_axxy_flush - hand_a_xxyy_flush)

    hand_a_axyz = 15 * comb(12, 3) * 16**3
    hand_a_hhyz = 3 * comb(16, 2) * comb(11, 2) * 16**2
    hand_a_axyz_flush = 3 * comb(12, 3) * 4**3
    hand_a_hhyz_flush = 3 * comb(4, 2) * comb(11, 2) * 4**2
    assert freq.jack_high_pair == (hand_a_axyz + hand_a_hhyz -
                                   hand_a_axyz_flush - hand_a_hhyz_flush)


def test_hand_freq_4deck_2same():
    freq = _run_hand_freq('AC AC', num_decks=4)
    assert freq.royal_flush == 0
    assert freq.straight_flush == 0
    assert freq.five_kind == comb(14, 3)  # AA+AAA
    assert freq.four_kind == comb(14, 2) * 12 * 16  # AA+AAX

    hand_aa_axx = 14 * 12 * comb(16, 2)  # AA+AXX
    hand_aa_xxx = 12 * comb(16, 3)  # AA+XXX
    assert freq.full_house == hand_aa_axx + hand_aa_xxx

    hand_aa_xyz_flush = comb(12, 3) * 4**3
    hand_aa_xxy_flush = 12 * comb(4, 2) * 11 * 4
    hand_aa_axy_flush = 2 * comb(12, 2) * 4**2
    assert freq.flush == (hand_aa_xyz_flush + hand_aa_xxy_flush +
                          hand_aa_axy_flush)

    hand_aa_axy = 14 * comb(12, 2) * 16**2
    hand_aa_axy_flush = 2 * comb(12, 2) * 4**2
    assert freq.three_kind == hand_aa_axy - hand_aa_axy_flush

    hand_aa_xxy = 12 * comb(16, 2) * 11 * 16
    hand_aa_xxy_flush = 12 * comb(4, 2) * 11 * 4
    assert freq.two_pair == hand_aa_xxy - hand_aa_xxy_flush

    hand_aa_xyz = comb(12, 3) * 16**3
    hand_aa_xyz_flush = comb(12, 3) * 4**3
    assert freq.jack_high_pair == hand_aa_xyz - hand_aa_xyz_flush


def test_hand_freq_royal_flush_4deck_2different():
    freq = _run_hand_freq('AC 2C', num_decks=4)
    assert freq.royal_flush == 0
    assert freq.straight_flush == comb(4, 1)**3  # A2+345 flush
    assert freq.five_kind == 0
    assert freq.four_kind == 2 * comb(15, 3)  # A2+222 or A2+AAA
    assert freq.full_house == 2 * 15 * comb(15, 2)  # A2+A22 or A2+AA2

    hand_a2_xyz_flush = comb(11, 3) * 4**3  # all different rank
    hand_a2_axy_flush = 3 * comb(11, 2) * 4**2  # pair Ace
    hand_a2_2xy_flush = 3 * comb(11, 2) * 4**2  # pair 2
    hand_a2_xxy_flush = 11 * comb(4, 2) * 10 * 4  # pair other
    hand_a2_a2x_flush = 3 * 3 * 11 * 4  # two pair: A, 2
    hand_a2_axx_flush = 3 * 11 * comb(4, 2)  # two pair: A, other
    hand_a2_2xx_flush = 3 * 11 * comb(4, 2)  # two pair: 2, other
    hand_a2_aax_flush = comb(3, 2) * 11 * 4  # three Ace
    hand_a2_22x_flush = comb(3, 2) * 11 * 4  # three two
    hand_a2_xxx_flush = 11 * comb(4, 3)  # three other
    hand_a2_345_flush = 4**3  # straight flush
    assert freq.flush == (hand_a2_xyz_flush + hand_a2_axy_flush +
                          hand_a2_2xy_flush + hand_a2_xxy_flush +
                          hand_a2_aax_flush + hand_a2_22x_flush +
                          hand_a2_axx_flush + hand_a2_2xx_flush +
                          hand_a2_a2x_flush + hand_a2_xxx_flush -
                          hand_a2_345_flush)

    hand_a2_345 = 16**3
    assert freq.straight == hand_a2_345 - hand_a2_345_flush

    hand_a2_aax = comb(15, 2) * 11 * 16
    hand_a2_22x = comb(15, 2) * 11 * 16
    hand_a2_xxx = 11 * comb(16, 3)
    assert freq.three_kind == (hand_a2_aax + hand_a2_22x + hand_a2_xxx -
                               hand_a2_aax_flush - hand_a2_22x_flush -
                               hand_a2_xxx_flush)


def test_best_expected_payout():
    hand_str = 'AS 2C 3D 4H 8S'
    shoe_str = 'AH 5C 7H JS KH'
    # Straight=10, jack-high-pair=5
    payout = payout_table.PayoutTable(0, 0, 0, 0, 0, 0, 10, 0, 0, 5, 0)
    test_hs = hand_stats.HandStats(card.card_list(shoe_str), payout=payout)
    best_hand, best_payout = test_hs.best_expected_payout(
        delt=hand.Hand(card.card_list(hand_str)))
    # Best is to go for a pair of aces: 80% * 5 = 4
    assert best_hand == hand.Hand(card.card_list('AS'))
    assert best_payout == 4


def test_best_expected_short_shoe():
    hand_str = 'AS 2C 3D 4H 8S'
    shoe_str = 'AH 5C'
    # Straight=10, jack-high-pair=5
    payout = payout_table.PayoutTable(0, 0, 0, 0, 0, 0, 10, 0, 0, 5, 0)
    test_hs = hand_stats.HandStats(card.card_list(shoe_str), payout=payout)
    held_values = test_hs.best_expected_payout_per_num_held(
        delt=hand.Hand(card.card_list(hand_str)))
    # Values for holding 0-2 cards should be zero
    expected = [hand_stats.HeldValue(hand.Hand(''), 0.0),
                hand_stats.HeldValue(hand.Hand('AS'), 0.0),
                hand_stats.HeldValue(hand.Hand('AS 2C'), 0.0),
                hand_stats.HeldValue(hand.Hand('2C 3D 4H'), 10.0),
                hand_stats.HeldValue(hand.Hand('AS 2C 3D 4H'), 7.5),
                hand_stats.HeldValue(hand.Hand('AS 2C 3D 4H 8S'), 0.0)]
    assert held_values == expected


def test_best_expected_payout_with_advantage():
    hand_str = 'AS 2C 3D 4H 8S'
    shoe_str = 'AH 5C 7H JS KH'
    # Straight=10, jack-high-pair=5
    payout = payout_table.PayoutTable(0, 0, 0, 0, 0, 0, 10, 0, 0, 5, 0)
    test_hs = hand_stats.HandStats(card.card_list(shoe_str), payout=payout)
    advantage = [-10, -10, -10, -10, 40, -10]
    best_hand, best_payout = test_hs.best_expected_payout(
        delt=hand.Hand(card.card_list(hand_str)), advantage=advantage)
    # With net +50 advantage to holding four cards the new recommendation is to
    # go for the inside straight: 0.2 * 10 + 0.2 * 5 = 3
    assert best_hand == hand.Hand(card.card_list('AS 2C 3D 4H'))
    assert best_payout == 3


def test_best_expected_payout_risk_averse():
    hand_str = 'AS 2C 3D 4H 8S'
    shoe_str = 'AH 5C 7H JS KH'
    # Straight=10, jack-high-pair=5
    payout = payout_table.PayoutTable(0, 0, 0, 0, 0, 0, 10, 0, 0, 5, 0)
    test_hs = hand_stats.HandStats(card.card_list(shoe_str), payout=payout)
    best_hand, best_payout = test_hs.best_expected_payout(
        delt=hand.Hand(card.card_list(hand_str)), risk=-1)
    # Best is still to hold the Ace for pair of aces: E[R] = 80% * 5 = 4
    # Downside risk adjustment = -1 * sqrt(.2 * (0-4)**2) = -1.7889
    assert best_hand == hand.Hand(card.card_list('AS'))
    npt.assert_almost_equal(best_payout, 2.2111456)


def test_best_expected_payout_risk_seeking():
    hand_str = 'AS 2C 3D 4H 8S'
    shoe_str = 'AH 5C 7H JS KH'
    # Straight=10, jack-high-pair=5
    payout = payout_table.PayoutTable(0, 0, 0, 0, 0, 0, 10, 0, 0, 5, 0)
    test_hs = hand_stats.HandStats(card.card_list(shoe_str), payout=payout)
    best_hand, best_payout = test_hs.best_expected_payout(
        delt=hand.Hand(card.card_list(hand_str)), risk=1)
    # Best is to go for the outside straight: E[R] = 20% * 10 + 20% * 5 = 3
    # Upside risk adjustment = 1 * sqrt(.2 * (10-3)**2 + .2 * (5-3)**2) = 3.256
    assert best_hand == hand.Hand(card.card_list('AS 2C 3D 4H'))
    npt.assert_almost_equal(best_payout, 6.25576412)


def test_best_expected_payout_noise():
    hand_str = 'AS 2C 3D 4H 8S'
    shoe_str = 'AH 5C 7H JS KH'
    # Straight=10, jack-high-pair=5
    payout = payout_table.PayoutTable(0, 0, 0, 0, 0, 0, 10, 0, 0, 5, 0)
    test_hs = hand_stats.HandStats(card.card_list(shoe_str), payout=payout,
                                   rng=np.random.default_rng(seed=12345))
    best_hand, best_payout = test_hs.best_expected_payout(
        delt=hand.Hand(card.card_list(hand_str)), noise=1)
    # Best is to go for a pair of aces: 80% * 5 = 4, plus noise.
    assert best_hand == hand.Hand(card.card_list('AS'))
    npt.assert_almost_equal(best_payout, 5.26372845812911)


def test_semi_deviance():
    prob = [.2, .4, .6]
    values = [0, 10, 20]
    dsr = hand_stats.semi_deviation(values, prob, upside=False)
    usr = hand_stats.semi_deviation(values, prob, upside=True)
    # weighted mean = 16
    # downside risk = sqrt(.2 * (0 - 16)**2 + .4 * (10 - 16)**2 + .6) = 8.099
    # upside risk = sqrt(.6 * 4**2) = 3.098
    npt.assert_almost_equal(dsr, 8.0993827)
    npt.assert_almost_equal(usr, 3.0983867)
