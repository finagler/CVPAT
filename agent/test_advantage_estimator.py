import numpy as np
import numpy.testing as npt

from agent import advantage_estimator
from game import card


def test_shoe_to_unnormed_states_unlimited():
    # 20 Cards. Note that the expected output only has 11 rows because
    # we stop play when there are fewer than 10 card remaining before
    # the start of a hand.
    cards = card.card_list('AC 2C AC 3C 4C 6C 7C 8C 7C 10C ' +
                           'JC JC JC JC QC QC QC QC KC KC')
    states = advantage_estimator.shoe_to_unnormed_states(
        cards, max_hand_limit=0)
    z39 = [0] * 39
    max_hl = [advantage_estimator.MAX_HAND_LIMIT]
    expected = np.array([
        [[2, 1, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [1, 1, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 2] + z39 + max_hl,
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2] + z39 + max_hl],
        ]).astype(int)
    npt.assert_equal(states, expected)


def test_shoe_to_unnormed_states():
    # 20 Cards. Note that the expected output only has 11 rows because
    # we stop play when there are fewer than 10 card remaining before
    # the start of a hand.
    cards = card.card_list('AC 2C AC 3C 4C 6C 7C 8C 7C 10C ' +
                           'JC JC JC JC QC QC QC QC KC KC')
    states = advantage_estimator.shoe_to_unnormed_states(
        cards, max_hand_limit=1)
    z39 = [0] * 39
    expected = np.array([
        [[2, 1, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [1, 1, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 2] + z39 + [0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2] + z39 + [0]],
        [[2, 1, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [1, 1, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 1, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 2] + z39 + [1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2] + z39 + [1]],
        ]).astype(int)
    npt.assert_equal(states, expected)
