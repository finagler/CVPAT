"""Methods for computing best discard / value of different poker hands."""
from __future__ import annotations

import collections
import pickle

import numpy as np
from scipy import special
import typing
from numpy import typing as npt
from numpy.lib import stride_tricks

from game import card
from game import hand
from game import payout_table


HandFrequency = collections.namedtuple(
    'HandFrequency',
    ('royal_flush', 'straight_flush', 'five_kind', 'four_kind', 'full_house',
     'flush', 'straight', 'three_kind', 'two_pair', 'jack_high_pair',
     'low_pair', 'no_hand'))


# Tuple of cards being held (Hand) and the expected payout after drawing back
# up to five cards (float)
HeldValue = collections.namedtuple('HeldValue', ('held', 'value'))


def canonical_hands_dict() -> dict:
    """Returns dict of canonical hands to number of hands that map to it.

    The resulting dict will contain 134,459 canonical hands, each mapping
    to a number of hands that map to it ranging between 4 and 24. The total
    number of values will sum to (52 choose 5) = 2,598,960.

    A cached value from this routine is saved at data/canonical_hands_dict.pkl.
    """
    canonical_dict = collections.defaultdict(int)
    it = hand.all_hands_iter()
    for h in it:
        canonical_dict[h.canonical_suit()] += 1
    return canonical_dict


def semi_deviation(values: npt.ArrayLike,
                   prob: npt.ArrayLike,
                   axis: typing.Optional[int] = None,
                   upside: bool = False):
    """Returns semi-deviation as a measure of downside (or upside) risk.

    By default, returns downside risk, defined as:
        sd = sqrt(sum(prob * min(values - E[values], 0) ** 2))
    If upside is True then returns upside risk:
        sd = sqrt(sum(prob * max(values - E[values], 0) ** 2))

    where E[values] = sum(prob * values)

    Args:
        values: values to compute risk for
        prob: probability (or weight) of each value
        axis: axis to compute along
        upside: if True then compute upside risk instead of downside.

    Returns: downside (or upside) risk.
    """
    values = np.array(values)
    prob = np.array(prob)
    expected = np.sum(values * prob, axis=axis, keepdims=True)
    if upside:
        deviations = np.maximum(values - expected, 0)
    else:
        deviations = np.minimum(values - expected, 0)
    return np.sqrt(np.sum(prob * deviations ** 2, axis=axis))


def _k_kind_freq(have: npt.ArrayLike, held: hand.Hand) -> npt.NDArray[np.int64]: # np.ndarray:
    """Returns number of ways to draw K-of-a-kind type hands, ignoring flushes.

    Returns the number of unique ways to form five-of-a-kind, four-of-a-kind,
    full house, three-of-a-kind, two pair, jack-high pair and low pair given
    the number of cards of each rank held in the shoe ('have') and cards being
    held. Counts do not take suit into account, so values for three-of-a-kind,
    two pair and jack-high pair will also include combinations that would
    instead count as a flush.

    Args:
        have: number of each rank in shoe (Ace through King)
        held: cards being held in hand.

    Returns:
         [freq_5kind, freq_4kind, freq_full_house, freq_3kind, freq_2pair,
         freq_jack_high_pair, freq_low_pair]: array of number of unique
         combinations that form each hand type (ignoring flush).
    """
    have = np.array(have)
    empty_slots = held.num_empty_slots()

    # How many additional cards we need of each rank to make match. Shape: (13,)
    need_singleton = 1 - held.num_ranks
    need_pair = 2 - held.num_ranks
    need_triplet = 3 - held.num_ranks
    need_quad = 4 - held.num_ranks
    need_quint = 5 - held.num_ranks

    # Number of ways to form match of given number for each rank. This does not
    # count additional slots (e.g. ways_pair is only ways to form two cards
    # of the given rank). Shape: (13,)
    ways_singleton = special.binom(have, need_singleton).astype(int)
    ways_pair = special.binom(have, need_pair).astype(int)
    ways_triplet = special.binom(have, need_triplet).astype(int)
    ways_quad = special.binom(have, need_quad).astype(int)
    ways_quint = special.binom(have, need_quint).astype(int)

    # Five-of-a-kind.
    ways_5kind = np.where(need_quint <= empty_slots, ways_quint, 0)
    freq_5kind = ways_5kind.sum()

    # Four-of-a-kind is quad + singleton of different ranks.
    ways_4kind = ways_quad[:, np.newaxis] * ways_singleton[np.newaxis, :]
    np.fill_diagonal(ways_4kind, 0)
    # Eliminate combinations that require too many slots.
    need_4kind = need_quad[:, np.newaxis] + need_singleton[np.newaxis, :]
    ways_4kind = np.where(need_4kind <= empty_slots, ways_4kind, 0)
    freq_4kind = ways_4kind.sum()

    # Full house is a triplet + pair of different ranks.
    ways_fh = ways_triplet[:, np.newaxis] * ways_pair[np.newaxis, :]
    np.fill_diagonal(ways_fh, 0)
    # Eliminate combinations that require too many slots.
    need_fh = need_triplet[:, np.newaxis] + need_pair[np.newaxis, :]
    ways_fh = np.where(need_fh <= empty_slots, ways_fh, 0)
    freq_full_house = ways_fh.sum()

    # Three-of-a-kind is a triplet + two singletons, all of different ranks.
    # Since the two singletons are undifferentiated we also exclude cases
    # where rank(S2) <= rank(S1), to avoid double-counting.
    ways_3kind = (ways_triplet[:, np.newaxis, np.newaxis] *
                  ways_singleton[np.newaxis, :, np.newaxis] *
                  ways_singleton[np.newaxis, np.newaxis, :])
    zero_diagonal = np.ones((13, 13)).astype(int)
    np.fill_diagonal(zero_diagonal, 0)
    ways_3kind *= zero_diagonal[:, :, np.newaxis]  # T != S1
    ways_3kind *= zero_diagonal[:, np.newaxis, :]  # T != S2
    tri_upper = np.triu(np.ones((13, 13)), k=1).astype(int)
    ways_3kind *= tri_upper[np.newaxis, :, :]  # S1 < S2
    # Eliminate combinations that require too many slots.
    need_3kind = (need_triplet[:, np.newaxis, np.newaxis] +
                  need_singleton[np.newaxis, :, np.newaxis] +
                  need_singleton[np.newaxis, np.newaxis, :])
    ways_3kind = np.where(need_3kind <= empty_slots, ways_3kind, 0)
    freq_3kind = ways_3kind.sum()

    # Two pair is two pair and a singleton, all of different ranks.
    # Since the two pair are undifferentiated we need to exclude cases where
    # rank(P2) <= rank(P1), to avoid double-counting.
    ways_2pair = (ways_singleton[:, np.newaxis, np.newaxis] *
                  ways_pair[np.newaxis, :, np.newaxis] *
                  ways_pair[np.newaxis, np.newaxis, :])
    ways_2pair *= zero_diagonal[:, :, np.newaxis]  # S != P1
    ways_2pair *= zero_diagonal[:, np.newaxis, :]  # S != P2
    ways_2pair *= tri_upper[np.newaxis, :, :]  # P1 < P2
    # Eliminate combinations that require too many slots.
    need_2pair = (need_singleton[:, np.newaxis, np.newaxis] +
                  need_pair[np.newaxis, :, np.newaxis] +
                  need_pair[np.newaxis, np.newaxis, :])
    ways_2pair = np.where(need_2pair <= empty_slots, ways_2pair, 0)
    freq_2pair = ways_2pair.sum()

    def _pair_freq(ways_1pair_4d: np.ndarray):
        """Returns frequency of one pair. Modifies ways_1pair.

        Args:
            ways_1pair_4d: 4d product of ways to generate a pair plus three
                singletons. Shape (13, 13, 13, 13).
        """
        # Pair rank different from singletons
        ways_1pair_4d *= zero_diagonal[:, :, np.newaxis, np.newaxis]  # P != S1
        ways_1pair_4d *= zero_diagonal[:, np.newaxis, :, np.newaxis]  # P != S2
        ways_1pair_4d *= zero_diagonal[:, np.newaxis, np.newaxis, :]  # P != S3
        # S1 < S2 < S3
        ways_1pair_4d *= tri_upper[np.newaxis, :, :, np.newaxis]  # S1 < S2
        ways_1pair_4d *= tri_upper[np.newaxis, np.newaxis, :, :]  # S2 < S3
        # Eliminate combinations that require too many slots.
        need_1pair = (
                need_pair[:, np.newaxis, np.newaxis, np.newaxis] +
                need_singleton[np.newaxis, :, np.newaxis, np.newaxis] +
                need_singleton[np.newaxis, np.newaxis, :, np.newaxis] +
                need_singleton[np.newaxis, np.newaxis, np.newaxis, :])
        ways_1pair_4d = np.where(need_1pair <= empty_slots, ways_1pair_4d, 0)
        return ways_1pair_4d.sum()

    ways_1pair = (
            ways_pair[:, np.newaxis, np.newaxis, np.newaxis] *
            ways_singleton[np.newaxis, :, np.newaxis, np.newaxis] *
            ways_singleton[np.newaxis, np.newaxis, :, np.newaxis] *
            ways_singleton[np.newaxis, np.newaxis, np.newaxis, :])
    # Mask out to make Jack-high and low-pair versions.
    ways_pair_jack_high = ways_1pair.copy()
    ways_pair_jack_high[1:-3, :, :, :] = 0
    ways_1pair[0, :, :, :] = 0
    ways_1pair[-3:, :, :, :] = 0

    freq_pair_jack_high = _pair_freq(ways_pair_jack_high)
    freq_1pair = _pair_freq(ways_1pair)

    return np.array([freq_5kind, freq_4kind, freq_full_house,
                     freq_3kind, freq_2pair, freq_pair_jack_high,
                     freq_1pair])


def get_card_count(cards: typing.Iterable[card.Card]) -> np.ndarray:
    """Returns number of each unique card in list, as array of shape (4, 13).

    Args:
        cards: iterable of cards

    Returns: count, where count[suit_index, rank_index] is the number of
        instances of that card.
    """
    count = np.zeros((4, 13)).astype(int)
    for c in cards:
        count[c.suit_index, c.rank_index] += 1
    return count


class HandStats(object):
    """Object that computes probability of winning hands given a shoe of cards.
    
    A HandStats instance computes the probability of getting various winning
    hands (or the expected value of such hand) given a five-card hand and a
    shoe.

    Attributes:
        payout: PayoutTable

    Args:
        cards: cards in shoe (order does not matter).
        payout: payout table to use.
    """
    def __init__(self, cards: typing.Iterable[card.Card],
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 rng: typing.Optional[np.random.Generator] = None):
        """Initialize HandStats object.

        Args:
            cards: cards in the shoe. Order is ignored.
            payout: the payout table in use, or None for default.
            rng: random number generator to use for computing noise (if
                requested), or None for default rng.
        """
        self.payout = payout or payout_table.PayoutTable.default()
        self._cards = get_card_count(cards)
        self._rng = rng or np.random.default_rng()

    @property
    def cards(self) -> np.ndarray:
        """Number of each card in shoe, indexed by [suit, rank]."""
        return self._cards

    @property
    def num_ranks(self) -> np.ndarray:
        """Number of each rank, as a 13-integer numpy array with Ace low."""
        return self._cards.sum(axis=0)

    @property
    def num_suits(self) -> np.ndarray:
        """Number of each suit, indexed Clubs, Diamonds, Hearts, Spades."""
        return self._cards.sum(axis=1)

    def _straight_type_freq(
            self, held: hand.Hand) -> typing.Tuple[int, int, int]:
        """Returns number ways to draw royal flush, straight flush and straight.
        
        Args:
            held: cards held in hand.

        Returns:
            (freq_royal_flush, freq_straight_flush, freq_straight)
        """
        def _royal_straight_freq(have: npt.ArrayLike) -> int:
            if held.has_partial_royal():
                to_use = np.atleast_2d(np.where(held.num_ranks, 1, have))
                return to_use[:, [0, 9, 10, 11, 12]].prod(axis=-1).sum()
            return 0

        def _ace_low_straight_freq(have: npt.ArrayLike) -> int:
            held_window = stride_tricks.sliding_window_view(held.num_ranks, 5)
            full_window_mask = held_window.sum(axis=-1) == held.size()
            have = np.atleast_2d(np.where(held.num_ranks, 1, have))
            have_window = stride_tricks.sliding_window_view(have, 5, axis=-1)[
                :, full_window_mask]
            return have_window.prod(axis=-1).sum()

        if not held.has_partial_straight():
            return 0, 0, 0

        if held.has_partial_flush():
            if held.empty():
                suit_mask = [True, True, True, True]
            else:
                suit_mask = (held.num_suits > 0)
            cards_in_suit = self._cards[suit_mask]  # Shape: (*, 13) 
            freq_royal_flush = _royal_straight_freq(cards_in_suit)
            freq_straight_flush = _ace_low_straight_freq(cards_in_suit)
        else:
            freq_royal_flush = 0
            freq_straight_flush = 0

        freq_royal_straight = _royal_straight_freq(self.num_ranks)
        freq_low_straight = _ace_low_straight_freq(self.num_ranks)
        freq_straight = (freq_royal_straight + freq_low_straight -
                         freq_royal_flush - freq_straight_flush)
        return freq_royal_flush, freq_straight_flush, freq_straight
    
    def hand_freq(self, held: hand.Hand) -> HandFrequency:
        """Returns number of ways to draw a winning hand given held cards.

        Computes the number of unique ways cards can be drawn to form each
        winning hand type.

        Args:
            held: cards being held.

        Returns:
            number of unique combinations that form each hand type
        """
        # Compute the ways to draw sequential-type hands.
        freq_royal_flush, freq_straight_flush, freq_straight = (
            self._straight_type_freq(held))

        (freq_5kind, freq_4kind, freq_full_house, freq_3kind, freq_2pair,
         freq_jack_high_pair, freq_low_pair) = _k_kind_freq(self.num_ranks,
                                                            held)

        if not held.has_partial_flush():
            freq_flush = 0
        else:
            # Compute number of ways to form each type of k-kind hand that is
            # also a flush.
            flush_freq_by_type = np.zeros(7).astype(int)

            if held.empty():
                suit_mask = [True, True, True, True]
            else:
                suit_mask = (held.num_suits > 0)
            cards_in_suit = self._cards[suit_mask]
            for row in cards_in_suit:
                flush_freq_by_type += _k_kind_freq(row, held)

            (flush_freq_5kind, flush_freq_4kind, flush_freq_full_house,
             flush_freq_3kind, flush_freq_2pair, flush_freq_jack_high_pair,
             flush_freq_low_pair) = flush_freq_by_type
            ways_flush = special.binom(cards_in_suit.sum(axis=-1),
                                       held.num_empty_slots()).astype(int)

            # Compute ways of forming a flush, and subtract out the ways
            # higher-ranked hands can also be a flush.
            freq_flush = (ways_flush.sum() - freq_royal_flush -
                          freq_straight_flush - flush_freq_5kind -
                          flush_freq_4kind - flush_freq_full_house)
            freq_3kind -= flush_freq_3kind
            freq_2pair -= flush_freq_2pair
            freq_jack_high_pair -= flush_freq_jack_high_pair
            freq_low_pair -= flush_freq_low_pair

        freq_no_hand = (self.freq_all_draws(held) - freq_royal_flush -
                        freq_straight_flush - freq_5kind - freq_4kind -
                        freq_full_house - freq_flush - freq_straight -
                        freq_3kind - freq_2pair - freq_jack_high_pair -
                        freq_low_pair)

        return HandFrequency(
            freq_royal_flush, freq_straight_flush, freq_5kind, freq_4kind,
            freq_full_house, freq_flush, freq_straight, freq_3kind, freq_2pair,
            freq_jack_high_pair, freq_low_pair, freq_no_hand)

    def size(self):
        return self._cards.sum()

    def freq_winning_hands(self, held: hand.Hand) -> np.ndarray:
        """Returns number of unique possible winning hands of each type."""
        return np.array(self.hand_freq(held))

    def freq_all_draws(self, held: hand.Hand) -> int:
        """Returns number of unique combinations drawing up to five cards."""
        return special.comb(self.size(), held.num_empty_slots(), exact=True)

    def prob_winning_hand(self, held: hand.Hand) -> np.ndarray:
        """Returns array of probabilities of winning each type of hand."""
        freq_win = self.freq_winning_hands(held)
        freq_draws = self.freq_all_draws(held)
        # Return 0 if not enough cards to draw to full hand.
        probs = np.divide(
            freq_win, freq_draws, out=np.zeros_like(freq_win, dtype=float),
            where=~np.isclose(
                freq_draws, np.zeros_like(freq_draws, dtype=float)))
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        return probs

    def expected_payout(
            self, held: hand.Hand, risk: float = 0, noise: float = 0.) -> float:
        """Returns expected payout given shoe and cards being held.

        Args:
            held: partial hand being held (zero to five cards).
            risk: risk sentiment, in range [-1.0, 1.0]. Negative penalizes
                payouts with higher downside risk, positive adds to payouts
                with upside risk.
            noise: gaussian noise to add to each prospective action's payout,
                expressed as a standard deviation.
        """
        prob = self.prob_winning_hand(held)
        values = self.payout.table
        reward = (prob * values).sum()
        adj = 0
        if risk != 0:
            adj = risk * semi_deviation(values, prob, upside=(risk > 0))
        if noise != 0:
            adj += self._rng.normal(scale=noise)
        return reward + adj

    def _expected_payout_with_replacement_iter(
            self, delt: hand.Hand,
            risk: float = 0, noise: float = 0.) -> typing.Iterator[HeldValue]:
        """Yields expected payout for all 32 possible discards."""
        for h in delt.discard_combinations_iter():
            yield HeldValue(h, self.expected_payout(h, risk, noise))


    def best_expected_payout_per_num_held(
            self, delt: hand.Hand, risk: float = 0,
            noise: float = 0.) -> typing.List[HeldValue]:
        """Returns cards to hold and max expected payout for number cards held.

        Returns list of cards to hold to maximize the risk-adjusted payout for
        a given number of cards held, and the expected risk-adjusted payout.

        Args:
            delt: cards currently in hand. This will normally be a full five
                card hand, though it's not checked.
            risk: risk tolerance in range [-1.0, 1.0]. If risk is negative
                then payouts with higher downside risk are penalized; if risk
                is positive then payouts with higher upside risk are given a
                bonus.
            noise: gaussian noise to add to each prospective action's payout,
                expressed as a standard deviation.

        Returns:
            best: list of best expected payouts assuming that a given number of
                cards is held. Specifically, best[num_held].value is the best
                expected payout when holding back num_held cards out of all
                possible combinations, and best[num_held].held is the hand
                that achieves that expected payout.
        """
        best = [HeldValue(None, 0.0)] * 6
        for hv in self._expected_payout_with_replacement_iter(
                delt, risk, noise):
            num_held = hv.held.size()
            if best[num_held].held is None or hv.value > best[num_held].value:
                best[num_held] = hv
        return best

    def best_expected_payout(
            self, delt: hand.Hand,
            advantage: typing.Optional[typing.Iterable[float]] = None,
            risk: float = 0,
            noise: float = 0.) -> HeldValue:
        """Returns expected payout and which hands to discard to get it.

        Args:
            delt: cards currently in hand. This will normally be a full five
                card hand, though it's not checked.
            advantage: offset to add to each 'cards held' slot (0-5).
            risk: risk tolerance in range [-1.0, 1.0]. If risk is negative
                then payouts with higher downside risk are penalized; if risk
                is positive then payouts with higher upside risk are given a
                bonus.
            noise: gaussian noise to add to each prospective action's payout,
                expressed as a standard deviation.

        Returns:
            (best_held_hand, best_expected_payout)
        """
        best = self.best_expected_payout_per_num_held(delt, risk, noise)
        best_payout = np.array([b.value for b in best])
        if advantage is None:
            argmax = np.argmax(best_payout)
        else:
            best_payout_plus_adv = np.array(advantage) + best_payout
            argmax = np.argmax(best_payout_plus_adv)
        return HeldValue(best[argmax].held, best_payout[argmax])
