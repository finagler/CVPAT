"""Class for computing optimal play-through given full knowledge of shoe."""
from __future__ import annotations

import collections
import typing

import numpy as np

from game import card
from game import hand
from game import payout_table
from stats import hand_stats


class PsychicSolver(object):
    """Class for computing optimal play-through given full knowledge of shoe.

    Attributes:
        hand_limit: number of hands to play, or zero for unlimited. Player
            is assumed to play until either hand_limit hands are played or
            shoe is depleted.
        payout: payout table to use to compute winnings.
        best_expected_only: if True then limit the choice of which cards to
            hold to the six possible choices suggested by HandStats, i.e. the
            cards that maximize the *expected* payout when holding between
            zero and five cards. The total payout will be an upper bound on
            the payout that can be achieved by any policy that maximizes
            expected payout. If False then the total payout will be the upper
            bound on any possible policy for a given shoe.
        allow_short_shoe: if True then allow play to end of the shoe,
            limiting replacements to remaining cards. If False then game
            ends when shoe has fewer than 10 cards before drawing new hand.
    """
    def __init__(self, cards: typing.Union[str, typing.Iterable[card.Card]],
                 hand_limit: int = 0,
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 best_expected_only: bool = False,
                 allow_short_shoe: bool = False):
        if isinstance(cards, str):
            self.cards = card.card_list(cards)
        else:
            self.cards = list(cards)
        self.hand_limit = hand_limit
        self.payout = payout or payout_table.PayoutTable.default()
        self.best_expected_only = best_expected_only
        self.allow_short_shoe = allow_short_shoe

    def _possible_discard_combinations(
            self, delt: hand.Hand,
            shoe_start_index: int) -> typing.List[hand.Hand]:
        """Returns list of allowable hands after discarding.

        If best_expected_only is False then this is simply all 32 possible
        ways to discard cards given the delt hand. Otherwise it's the six
        hands that would maximize the expected payoff given the contents of
        the remaining shoe (assuming unknown order).

        Args:
            delt: 5-card hand currently delt.
            shoe_start_index: index of the card after the current hand.

        Returns: list of possible hands.
        """
        if self.best_expected_only:
            hs = hand_stats.HandStats(cards=self.cards[shoe_start_index:],
                                      payout=self.payout)
            best_expected = hs.best_expected_payout_per_num_held(delt)
            return [hv.held for hv in best_expected]
        return [h for h in delt.discard_combinations_iter()]

    def _best_payout_per_num_held(
            self, possible_hands: typing.List[hand.Hand],
            replacements: typing.List[card.Card]) -> [hand_stats.HeldValue]:
        """Returns best payout and best held cards for each number held.

        Args:
            possible_hands: list of possible hands after discarding.
            replacements: up to five cards for replacements, in draw order.

        Returns:
            best: list of best actual payouts assuming that a given number of
                cards is held. Specifically, best[num_held].value is the best
                payout when holding back num_held cards out of all possible
                combinations, and best[num_held].held is the hand that achieves
                that payout.
        """
        best = [hand_stats.HeldValue(None, 0.0)] * 6
        for held in possible_hands:
            num_slots = held.num_empty_slots()
            filled = held.add(replacements[:num_slots])
            payout = filled.payout(self.payout)
            num_held = held.size()
            if best[num_held].held is None or payout > best[num_held].value:
                best[num_held] = hand_stats.HeldValue(held, payout)
        return best

    def _last_index(self):
        if self.allow_short_shoe:
            return len(self.cards) - 5
        return len(self.cards) - 10

    def _payouts_by_num_held(self) -> typing.Tuple[
            np.ndarray, typing.List[typing.List[hand.Hand]]]:
        """Returns max one-hand payout at each point in shoe for number held.

        Returns an array where each element is the best payout for a single hand
        starting at a given point in the shoe and holding a given number of
        cards (discarding and replacing the rest).

        Returns:
            payouts: ndarray of shape (shoe_size, 6), where payout[i, h] is
                the best payout of the 5-card hand from shoe.cards[i:i+5] after
                discarding and replacing (5-h) cards.
            held: 2d list of Hands, where held[i][h] is the hand that should
                be held when discarding (5-h) cards.
        """
        last_index = self._last_index()
        payouts = np.zeros((len(self.cards), 6))
        held = [None] * len(self.cards)
        for i in range(last_index + 1):
            delt = hand.Hand(self.cards[i:i+5])
            replacements = self.cards[i+5:i+10]
            possible_hands = self._possible_discard_combinations(
                delt, shoe_start_index=i+5)
            best = self._best_payout_per_num_held(possible_hands, replacements)
            payouts[i, :] = np.array([b.value for b in best])
            held[i] = [b.held for b in best]
        return np.array(payouts), held

    def _solve_all_unlimited(self) -> typing.Tuple[
            np.ndarray, np.ndarray, typing.List[typing.List[hand.Hand]]]:
        """Returns optimal play for all start positions with no hand limit.

        Returns array of maximum possible payouts for each possible start
        position in the shoe and max number of hands, assuming perfect knowledge
        of the shoe card order. It's assumed that the player must play until
        either there are no more cards in the shoe or the hand limit is reached.
        The player may only discard up to the number of cards left in the shoe,
        so they may not be able to discard all five cards on the last hand.

        Returns:
            values: ndarray of float where values[i] is the total payout with
                best play starting with the hand that starts at the ith card in
                the shoe and playing up to hand_limit hands (or to the end of
                the shoe, whichever comes first).
            num_held: ndarray of int with same shape as values, where
                num_held[i] is the number of cards that should be
                held to get the payout returned in values. The specific hand
                to hold is held[i][num_held[i].
            held: 2d list of Hands, where held[i][h] is the hand that should
                be held when discarding (5-h) cards.
        """
        shoe_size = len(self.cards)
        payouts, held = self._payouts_by_num_held()
        # pad values with six extra rows of zeros.
        values = np.zeros(shoe_size+6).astype(float)
        num_held = np.zeros(shoe_size).astype(int)
        last_index = self._last_index()
        for i in range(last_index, -1, -1):
            # q_values is the payout for number held plus the value of
            # the state we would get to (reversed since holding fewer cards
            # puts us further into the deck).
            q_values = payouts[i, :] + values[i+10:i+4:-1]
            best_held = np.argmax(q_values)
            values[i] = q_values[best_held]
            num_held[i] = best_held
        return values[:shoe_size], num_held, held

    def _solve_all_with_hand_limit(self) -> typing.Tuple[
            np.ndarray, np.ndarray, typing.List[typing.List[hand.Hand]]]:
        """Returns optimal play for all start positions and hand limits.

        Returns array of maximum possible payouts for each possible start
        position in the shoe and max number of hands, assuming perfect knowledge
        of the shoe card order. It's assumed that the player must play until
        either there are no more cards in the shoe or the hand limit is reached.
        The player may only discard up to the number of cards left in the shoe,
        so they may not be able to discard all five cards on the last hand.

        Returns:
            values: ndarray of float where values[hands_remaining-1, i] is the
                total payout with best play starting at the ith card in the shoe
                and playing up to hand_limit hands (or to the end of the shoe,
                whichever comes first). Shape: (hand_limit, shoe_size).
            num_held: ndarray of int with same shape as values, where
                num_held[hands_remaining-1, i] is the number of cards that
                should be held to get the payout returned in values. The
                specific hand to hold is held[i][num_held[hands_remaining-1, i].
            held: 2d list of Hands, where held[i][h] is the hand that should
                be held when discarding (5-h) cards.
        """
        shoe_size = len(self.cards)
        payouts, held = self._payouts_by_num_held()
        values = np.zeros((self.hand_limit+1, shoe_size+6)).astype(float)
        num_held = np.zeros((self.hand_limit+1, shoe_size)).astype(int)
        last_index = self._last_index()
        for hand_remaining in range(1, self.hand_limit + 1):
            for i in range(last_index, -1, -1):
                # q_values is the payout for number held plus the value of
                # the state we would get to (reversed since holding fewer cards
                # puts us further into the deck).
                q_values = payouts[i, :] + values[hand_remaining-1, i+10:i+4:-1]
                best_held = np.argmax(q_values)
                values[hand_remaining, i] = q_values[best_held]
                num_held[hand_remaining, i] = best_held
        return values[1:, :shoe_size], num_held[1:, :], held

    def solve_all(self) -> typing.Tuple[
            np.ndarray, np.ndarray, typing.List[typing.List[hand.Hand]]]:
        """Returns optimal play for all start positions and optional hand limit.

        Returns array of maximum possible payouts for each possible start
        position in the shoe and max number of hands, assuming perfect knowledge
        of the shoe card order. It's assumed that the player must play until
        either there are no more cards in the shoe or the hand limit is reached.
        The player may only discard up to the number of cards left in the shoe,
        so they may not be able to discard all five cards on the last hand.

        Returns:
            values: ndarray of float where values[i] is the total payout with
                best play starting with the hand starting at the ith card and
                playing up to hand_limit hands (or to the end of the shoe,
                whichever comes first).
            num_held: ndarray of int with same shape as values, where
                num_held[i] is the number of cards that should be
                held to get the payout returned in values. The specific hand
                to hold is held[i][num_held[i].
            held: 2d list of Hands, where held[i][h] is the hand that should
                be held when discarding (5-h) cards.
        """
        if self.hand_limit:
            return self._solve_all_with_hand_limit()
        return self._solve_all_unlimited()

    def solve(self) -> typing.Tuple[float, typing.List[hand.Hand]]:
        """Returns optimal play from start of shoe for my hand limit.

        Returns which cards to hold each hand to maximize the total payout given
        a shoe, along with the total payout. This uses the full knowledge of the
        shoe's card order and total_payout is the maximum possible payout for
        the shoe.

        Returns:
            total_payout: total payout for entire game.
            held: list of cards that should be held for each hand.
        """
        values, num_held, held = self.solve_all()
        if self.hand_limit:
            values = values[-1]
            num_held = num_held[-1]
        total_payout = values[0]
        held_q = collections.deque()
        idx = 0
        while (idx < num_held.shape[0] and
               (self.hand_limit == 0 or len(held_q) < self.hand_limit)):
            num_held_this_hand = num_held[idx]
            try:
                held_q.append(held[idx][num_held_this_hand])
            except TypeError:
                held_q.append(None)
            idx += 5 + (5 - num_held_this_hand)
        return total_payout, list(held_q)
