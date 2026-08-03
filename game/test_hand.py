import numpy.testing as npt

from game import card
from game import hand


def test_discard_iter():
    c1 = card.Card(1, card.SuitEnum.CLUB)
    c2 = card.Card(2, card.SuitEnum.CLUB)
    c3 = card.Card(3, card.SuitEnum.CLUB)
    c4 = card.Card(4, card.SuitEnum.CLUB)
    c5 = card.Card(5, card.SuitEnum.CLUB)
    cards = [c1, c2, c3, c4, c5]
    held = hand.Hand(cards)
    all_hands = [h for h in held.discard_combinations_iter()]
    assert len(all_hands) == 32
    assert all_hands == [
        hand.Hand(),
        hand.Hand([c1]),
        hand.Hand([c2]),
        hand.Hand([c3]),
        hand.Hand([c4]),
        hand.Hand([c5]),
        hand.Hand([c1, c2]),
        hand.Hand([c1, c3]),
        hand.Hand([c1, c4]),
        hand.Hand([c1, c5]),
        hand.Hand([c2, c3]),
        hand.Hand([c2, c4]),
        hand.Hand([c2, c5]),
        hand.Hand([c3, c4]),
        hand.Hand([c3, c5]),
        hand.Hand([c4, c5]),
        hand.Hand([c1, c2, c3]),
        hand.Hand([c1, c2, c4]),
        hand.Hand([c1, c2, c5]),
        hand.Hand([c1, c3, c4]),
        hand.Hand([c1, c3, c5]),
        hand.Hand([c1, c4, c5]),
        hand.Hand([c2, c3, c4]),
        hand.Hand([c2, c3, c5]),
        hand.Hand([c2, c4, c5]),
        hand.Hand([c3, c4, c5]),
        hand.Hand([c1, c2, c3, c4]),
        hand.Hand([c1, c2, c3, c5]),
        hand.Hand([c1, c2, c4, c5]),
        hand.Hand([c1, c3, c4, c5]),
        hand.Hand([c2, c3, c4, c5]),
        hand.Hand([c1, c2, c3, c4, c5])]


def test_num_empty_slots():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.CLUB),
             card.Card(3, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert held.num_empty_slots() == 2


# Test is_*
def test_is_royal_flush_only():
    held = hand.Hand('AC KC QC JC 10C')
    npt.assert_equal(held.eval_hand(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_is_straight_flush_only():
    held = hand.Hand('KC QC JC 10C 9C')
    npt.assert_equal(held.eval_hand(), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_is_5kind_only():
    held = hand.Hand('2C 2C 2C 2C 2C')
    npt.assert_equal(held.eval_hand(), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_is_4kind_only():
    held = hand.Hand('2C 2C 2C 2C KC')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])


def test_is_full_house_only():
    held = hand.Hand('2C 2C 2C KC KC')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])


def test_is_flush_only_with_3kind():
    held = hand.Hand('2C 4C KC KC KC')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])


def test_is_flush_only_with_2pair():
    held = hand.Hand('2C 4C 4C KC KC')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])


def test_is_flush_only_with_jack_high_pair():
    held = hand.Hand('2C 4C 9C KC KC')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])


def test_is_straight_only():
    held = hand.Hand('KD QC JC 10C 9C')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])


def test_is_3kind_only():
    held = hand.Hand('KC KD KH JC 9C')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])


def test_is_two_pair_only():
    held = hand.Hand('KC KD JC JC 9C')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])


def test_is_jack_high_pair_only():
    held = hand.Hand('JC JC 9C 5S 3H')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])


def test_is_low_pair_only():
    held = hand.Hand('10C 10C 9C 5S 3H')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])


def test_is_no_hand():
    held = hand.Hand('AC 2C 3C 4C 8S')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def test_is_no_hand_with_partial_hand():
    held = hand.Hand('AC 2C 3C 4C')
    npt.assert_equal(held.eval_hand(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


# Test possible_*
def test_everything_possible_with_empty():
    held = hand.Hand()
    assert held.possible_royal_flush()
    assert held.possible_straight_flush()
    assert held.possible_4kind()
    assert held.possible_flush()
    assert held.possible_full_house()
    assert held.possible_straight()
    assert held.possible_3kind()
    assert held.possible_2pair()
    assert held.possible_jack_high_pair()


# Royal Flush
def test_possible_royal_flush_1ace():
    cards = [card.Card(1, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert held.possible_royal_flush()


def test_possible_royal_flush_1two():
    cards = [card.Card(2, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert not held.possible_royal_flush()


# Straight Flush
def test_possible_straight_flush_1ace1five():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(5, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert held.possible_straight_flush()


def test_possible_straight_flush_1ace1six():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(6, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert not held.possible_straight_flush()


def test_possible_straight_flush_different_suit():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert not held.possible_straight_flush()


# 5kind
def test_possible_5kind_1card():
    cards = [card.Card(1, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert held.possible_5kind()


def test_possible_5kind_2same():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert held.possible_5kind()


def test_possible_5kind_2different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert not held.possible_5kind()


# 4kind
def test_possible_4kind_1card():
    cards = [card.Card(1, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert held.possible_4kind()


def test_possible_4kind_2same():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert held.possible_4kind()


def test_possible_4kind_2different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert held.possible_4kind()


def test_possible_4kind_3different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.DIAMOND),
             card.Card(5, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert not held.possible_4kind()


def test_possible_4kind_5kind():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.DIAMOND),
             card.Card(1, card.SuitEnum.DIAMOND),
             card.Card(1, card.SuitEnum.DIAMOND),
             card.Card(1, card.SuitEnum.DIAMOND)]
    held = hand.Hand(cards)
    assert not held.possible_4kind()


# Flush
def test_possible_flush_one_suit():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert held.possible_flush()


def test_possible_flush_2suits():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_flush()


# Straight
def test_possible_straight_within_spread():
    cards = [card.Card(2, card.SuitEnum.CLUB),
             card.Card(4, card.SuitEnum.HEART),
             card.Card(6, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.possible_straight()


def test_possible_straight_outside_spread():
    cards = [card.Card(2, card.SuitEnum.CLUB),
             card.Card(4, card.SuitEnum.HEART),
             card.Card(7, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_straight()


def test_possible_straight_low_spread_with_ace():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(3, card.SuitEnum.CLUB),
             card.Card(5, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.possible_straight()


def test_possible_straight_high_spread_with_ace():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(10, card.SuitEnum.CLUB),
             card.Card(12, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.possible_straight()


def test_possible_straight_outside_spread_with_ace():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(6, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert not held.possible_straight()


def test_possible_straight_with_full_house():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.DIAMOND),
             card.Card(1, card.SuitEnum.HEART),
             card.Card(5, card.SuitEnum.DIAMOND),
             card.Card(5, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_straight()


def test_possible_straight_with_straight_flush():
    cards = [card.Card(2, card.SuitEnum.CLUB),
             card.Card(3, card.SuitEnum.CLUB),
             card.Card(4, card.SuitEnum.CLUB),
             card.Card(5, card.SuitEnum.CLUB),
             card.Card(6, card.SuitEnum.CLUB)]
    held = hand.Hand(cards)
    assert not held.possible_straight()


# 3kind
def test_possible_3kind_3different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.HEART),
             card.Card(5, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.possible_3kind()


def test_possible_3kind_4different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.HEART),
             card.Card(4, card.SuitEnum.HEART),
             card.Card(10, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_3kind()


def test_possible_3kind_2pair():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.HEART),
             card.Card(5, card.SuitEnum.HEART),
             card.Card(5, card.SuitEnum.SPADE)]
    held = hand.Hand(cards)
    assert not held.possible_3kind()


def test_possible_3kind_3same_flush():
    cards = [card.Card(1, card.SuitEnum.HEART),
             card.Card(1, card.SuitEnum.HEART),
             card.Card(1, card.SuitEnum.HEART),
             card.Card(5, card.SuitEnum.HEART),
             card.Card(8, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_3kind()


# 2pair
def test_possible_2pair_3different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(5, card.SuitEnum.HEART),
             card.Card(8, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.possible_2pair()


def test_possible_2pair_4different():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(5, card.SuitEnum.HEART),
             card.Card(7, card.SuitEnum.HEART),
             card.Card(8, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_2pair()


# Jack-high pair
def test_possible_jack_high_pair_2pair():
    cards = [card.Card(11, card.SuitEnum.CLUB),
             card.Card(11, card.SuitEnum.HEART),
             card.Card(2, card.SuitEnum.HEART),
             card.Card(2, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_jack_high_pair()


def test_possible_jack_high_pair_2low():
    cards = [card.Card(10, card.SuitEnum.HEART),
             card.Card(10, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert not held.possible_jack_high_pair()


def test_possible_jack_high_pair_2jack():
    cards = [card.Card(11, card.SuitEnum.HEART),
             card.Card(11, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.possible_jack_high_pair()


def test_num_duplicates():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.HEART),
             card.Card(2, card.SuitEnum.HEART),
             card.Card(2, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.num_duplicates() == [2]


def test_num_duplicates_reverse_order():
    cards = [card.Card(1, card.SuitEnum.CLUB),
             card.Card(1, card.SuitEnum.CLUB),
             card.Card(2, card.SuitEnum.HEART),
             card.Card(2, card.SuitEnum.HEART),
             card.Card(2, card.SuitEnum.HEART)]
    held = hand.Hand(cards)
    assert held.num_duplicates() == [3, 2]


def test_canonical_suit():
    held = hand.Hand('QS 2S 5H 4H KC')
    expected = hand.Hand('QC 2C 5D 4D KH')
    assert held.canonical_suit() == expected
