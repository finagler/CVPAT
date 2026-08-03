import typing

import numpy as np
from numpy import typing as npt

from game import card


# STATE_DIM = 53  # 52 cards + hand limit
STATE_DIM = 19  # 13 ranks + 4 suits + 1 shoe size + 1 hand limit
NUM_ACTIONS = 6  # Number held cards 0 to 5
MAX_NUM_DECKS = 4
MAX_HAND_LIMIT = MAX_NUM_DECKS * 52 // 5 - 1
STATE_DTYPE = np.float32


class AdvantageEstimator(object):
    """Object that estimates advantage of holding cards.

    Estimates the relative advantage of holding a given number of cards,
    given the contents (but not order) of the remaining shoe. Specifically,
    adv = A(state) is a 6-float vector, where adv[h] is the relative
    additional advantage of holding h cards above and beyond the expected
    reward from that action.

    Base class simply returns a constant bias.
    """
    def __init__(self, advantage: npt.ArrayLike):
        self.advantage = np.array(advantage)

    def get_advantage(self, cards: np.ndarray, hand_limit: int = 0):
        return self.advantage


# class TableAdvantageEstimator(AdvantageEstimator):
#     """Advantage estimator that uses a lookup table.
#
#     Advantages are loaded from a table keyed by number of cards remaining in
#     shoe and hand limit (number of hands remaining after the current one).
#     """
#     def __init__(self, table: np.ndarray):
#         self.table = table
#
#     # def get_advantage(self, cards: np.ndarray, hand_limit: int = 0):
#     #     return self.advantage


def preprocess_state_vector(state: np.ndarray) -> np.ndarray:
    """Encode state vector for training.

    Output state vector encodes relative rank and suit and absolute number
    of cards remaining, all centered on zero and normalized to [-1, 1].
    Output will be of shape (..., 18), where for a single output row:
        out[..., :13] = fraction of remaining cards for each rank, scaled
            and translated to range [-1, 1].
        out[..., 13:17] = fraction of remaining cards for each suit, scaled
            and translated to range [-1, 1].
        out[..., 17] = number of cards remaining in shoe, scaled as fraction
            of MAX_NUM_DECKS * 52 and translated to range [-1, 1].
        out[..., 18] = number of hands remaining, scaled as fraction
            of MAX_HAND_LIMIT and translated to range [-1, 1].

    Args:
        state: array of 53 elements containing count of each unique card in
            shoe, in order of A♣-K♣, A♢-K♢, A♡-K♡, A♠-K♠, followed by number
            of hands remaining. Will be flattened before use.

    Returns: processed state vector
    """
    assert state.shape[-1] == 53
    hands_remaining = state[..., -1]
    num_cards = state.sum(axis=-1)
    out = np.zeros(state.shape[:-1] + (13 + 4 + 1 + 1,)).astype(STATE_DTYPE)
    state = state[..., :-1].reshape(state.shape[:-1] + (4, 13))
    out[..., :13] = state.sum(axis=-2)  # number ranks
    out[..., 13:13+4] = state.sum(axis=-1)  # number suits
    out[..., -1] = num_cards
    # Encode as proportion of cards left in shoe, in range [-1, 1].
    out[..., :13] = (out[..., :13] * 2 / num_cards[..., np.newaxis]) - 1.0
    out[..., 13:17] = (out[..., 13:17] * 2 / num_cards[..., np.newaxis]) - 1.0
    out[..., 17] = (out[..., 17] * 2 / num_cards) - 1.0
    out[..., 18] = (hands_remaining * 2 / MAX_HAND_LIMIT) - 1.0
    return out

def shoe_to_unnormed_states(
        cards: typing.List[card.Card], max_hand_limit: int) -> np.ndarray:
    """Returns array of state vectors for all positions in shoe and hand limits.

    Args:
        cards: list of cards in shoe, in order.
        max_hand_limit: max number of hands to include. A value of zero
            indicates unlimited, and will produce a single row of hand limits
            with limit set to MAX_HAND_LIMITS for all.

    Returns: states, an array of shape (hand_limits, len(cards) - 9, 53),
        where S = states[p] is the 53-int state vector for the hand starting
        at position p. S[suit * 13 + rank] is the number of instances of card
        with given suit and rank, and S[52] is the hand limit.
    """
    # Create one-hot array for each card at position p.
    shoe_size = len(cards)
    one_hots = np.zeros((shoe_size, 52)).astype(int)
    for p, c in enumerate(cards):
        one_hots[p, c.suit_index * 13 + c.rank_index] = 1
    cumulative_sum_r = one_hots[::-1].cumsum(axis=0)[::-1]
    states = np.zeros((max_hand_limit + 1, shoe_size, 53))
    states[:, :, :-1] = cumulative_sum_r[np.newaxis, :, :]
    if max_hand_limit == 0:
        states[:, :, -1] = MAX_HAND_LIMIT
    else:
        states[:, :, -1] = np.arange(0, max_hand_limit + 1)[:, np.newaxis]
    return states[:, :shoe_size-9, :]

