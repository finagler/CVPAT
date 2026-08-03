"""Class representing a (possibly partially-filled) five-card poker hand.

A hand can have between zero and five cards. Methods are provided for discarding
and drawing cards, for determining whether a hand counts as a particular poker
hand, and for determining whether it is possible to draw up to a given poker
hand given the cards already being held.
"""
from __future__ import annotations

import itertools

from game import card
from game import payout_table

import numpy as np
import typing


class Hand(object):
    """A hand of up to five cards."""

    def __init__(
            self,
            cards: typing.Union[None, str, typing.Iterable[card.Card]] = None):
        if isinstance(cards, str):
            cards = card.card_list(cards)
        elif cards is None:
            cards = []
        self._cards = sorted([c for c in cards])
        self.num_suits = np.zeros(4).astype(int)
        self.num_ranks = np.zeros(13).astype(int)
        for c in self.cards:
            self.num_suits[c.suit_index] += 1
            self.num_ranks[c.rank_index] += 1

    def __hash__(self):
        return hash(tuple(self.cards))

    def __eq__(self, other):
        return self.cards == other.cards

    def __repr__(self):
        return f'Hand({self.cards})'

    def __str__(self):
        return '['+''.join([c.__repr__() for c in self.cards])+']'

    @property
    def cards(self):
        return self._cards

    def add(self, cards: typing.List[card.Card]) -> Hand:
        """Returns new hand with cards added.

        Args:
            cards: list of cards to add

        Raises:
             ValueError: if cards would make hand larger than 5 cards.
        """
        if len(cards) + self.size() > 5:
            raise ValueError(
                f'Cannot add {len(cards)} cards to {self.size()}-card hand.')
        return Hand(self.cards + cards)

    def discard_combinations_iter(self) -> typing.Iterator[Hand]:
        """Yields all hands achievable by discarding, including itself."""
        for k in range(6):
            for cards in itertools.combinations(self.cards, k):
                yield Hand(cards)

    def size(self):
        return len(self.cards)

    def payout(self, payout: payout_table.PayoutTable) -> float:
        """Returns payout for given five-card hand."""
        return (self.eval_hand() * payout.table).sum()

    def empty(self):
        return len(self.cards) == 0

    def full(self):
        return len(self.cards) == 5

    def num_empty_slots(self) -> int:
        return 5 - len(self.cards)

    def num_unique_suits(self):
        return self.num_suits.astype(bool).sum()

    def num_unique_ranks(self):
        return self.num_ranks.astype(bool).sum()

    def num_pair(self):
        return (self.num_ranks == 2).sum()

    # Helper functions used below in the 'is_*' methods.
    def _has_flush(self) -> bool:
        """Returns True if has 5 cards of same suit."""
        return self.num_suits.max(initial=0) == 5

    def _has_straight(self) -> bool:
        """Returns True if has 5 cards in sequential order (including royal)."""
        return self.full() and (
                self._has_royal() or np.all(np.trim_zeros(self.num_ranks) == 1))

    def _has_royal(self) -> bool:
        """Returns True if has 10-J-Q-K-A."""
        return np.all(self.num_ranks[[0, 9, 10, 11, 12]] == 1)

    def _has_xkind(self, num_same: int) -> bool:
        """Returns True if has exactly num_same of some rank."""
        return np.any(self.num_ranks == num_same)

    def _has_at_least_xkind(self, num_same: int) -> bool:
        """Returns True if has at least num_same of some rank."""
        return np.any(self.num_ranks >= num_same)

    def _has_full_house(self) -> bool:
        """Returns True if hand is a full house."""
        return self._has_xkind(3) and self._has_xkind(2)

    # Functions that return whether hand is a given type. Takes into account
    # higher-ranking hands, e.g. 2C 2C 2C 9C JC is a flush, not three-of-a-kind.
    def is_royal_flush(self) -> bool:
        """Returns True if hand is a royal flush."""
        return self._has_flush() and self._has_royal()

    def is_straight_flush(self) -> bool:
        """Returns True if hand is a straight flush (and not higher)."""
        return (self._has_flush() and
                self._has_straight() and
                not self._has_royal())

    def is_5kind(self) -> bool:
        """Returns True if hand is five-of-a-kind."""
        return self._has_xkind(5)

    def is_4kind(self) -> bool:
        """Returns True if hand is a four of a kind (and not higher)."""
        return self._has_xkind(4)

    def is_full_house(self) -> bool:
        """Returns True if hand is a full house."""
        return self._has_full_house()

    def is_flush(self) -> bool:
        """Returns True if hand is a flush (and not higher)."""
        return (self._has_flush() and
                not self._has_straight() and
                not self._has_at_least_xkind(4) and
                not self._has_full_house())

    def is_straight(self) -> bool:
        """Returns True if hand is a straight (and not higher)."""
        return self._has_straight() and not self._has_flush()

    def is_3kind(self) -> bool:
        """Returns True if hand is a three of a kind (and not higher)."""
        return (self._has_xkind(3) and not self._has_full_house() and
                not self._has_flush())

    def is_2pair(self) -> bool:
        """Returns True if hand is two pair (and not higher)."""
        return self.num_pair() == 2 and not self._has_flush()

    def is_jack_high_pair(self) -> bool:
        """Returns True if hand is a pair of Jacks, Queens, Kings or Aces."""
        return (np.any(self.num_ranks[[0, 10, 11, 12]] == 2) and
                not self.num_pair() > 1 and
                not self._has_at_least_xkind(3) and
                not self._has_flush())

    def is_low_pair(self) -> bool:
        """Returns True if hand is a pair of Two through Ten."""
        return (np.any(self.num_ranks[1:10] == 2) and
                not self.num_pair() > 1 and
                not self._has_at_least_xkind(3) and
                not self._has_flush())

    def eval_hand(self) -> np.ndarray:
        value = np.array([self.is_royal_flush(),
                          self.is_straight_flush(),
                          self.is_5kind(),
                          self.is_4kind(),
                          self.is_full_house(),
                          self.is_flush(),
                          self.is_straight(),
                          self.is_3kind(),
                          self.is_2pair(),
                          self.is_jack_high_pair(),
                          self.is_low_pair(),
                          False])
        if not np.any(value):
            value[-1] = True
        assert value.sum() == 1
        return value

    def num_duplicates(self) -> typing.List[int]:
        """Returns list of counts of duplicate cards (rank and suit) in hand.

        Returns the counts of all sets of cards that match both rank and suit
        (i.e. twins, triplets, etc.) in the hand. The list is sorted in
        descending order of count, and ones are omitted. If no duplicates are
        found, returns an empty list.

        Example:
            [2C, 2C, 8S, 8S, JD] --> [2, 2]
            [KC, KC, KC, KS] --> [3]
            [AH 2D 2D AH 2D] --> [3, 2]"""
        counts = np.unique(self.cards, return_counts=True)[1]
        return sorted(counts[counts > 1])[::-1]

    # Helper functions used below in the 'possible_*' methods.
    def has_partial_royal(self) -> bool:
        """Returns True if has no ranks below 10."""
        return not np.any(self.num_ranks[1:9])

    def has_partial_flush(self) -> bool:
        """Returns True if has no more than one suit."""
        return self.num_unique_suits() <= 1

    def has_partial_straight(self) -> bool:
        """Returns True if cards ranks could possibly form a straight."""
        trimmed = np.trim_zeros(self.num_ranks)
        return (not self._has_at_least_xkind(2) and
                (len(trimmed) <= 5 or self.has_partial_royal()))

    # Functions that return whether a given hand type is possible given cards
    # already in hand (assuming a full 52-card deck apart from cards already
    # in hand). These will return False if the hand already makes a higher
    # ranked hand, e.g. possible_3kind() will return False for a hand that
    # already has four of a kind.
    def possible_royal_flush(self) -> bool:
        """Returns True if possible to draw to a royal flush."""
        return self.has_partial_flush() and self.has_partial_royal()

    def possible_straight_flush(self) -> bool:
        """Returns True if possible to draw to a straight flush."""
        return self.has_partial_straight() and self.has_partial_flush()

    def possible_5kind(self) -> bool:
        """Returns True if possible to draw to five-of-a-kind."""
        return self.num_unique_ranks() <= 1

    def possible_4kind(self) -> bool:
        """Returns True if possible to draw to four of a kind."""
        return (self.num_ranks.max(initial=0) + self.num_empty_slots() >= 4 and
                not self._has_xkind(5))

    def possible_full_house(self) -> bool:
        """Returns True if possible to draw to a full house."""
        return self.num_unique_ranks() < 3 and not self._has_at_least_xkind(4)

    def possible_flush(self) -> bool:
        """Returns True if possible to draw to a flush."""
        return (self.has_partial_flush() and
                not self.is_royal_flush() and
                not self.is_straight_flush() and
                not self._has_at_least_xkind(4) and
                not self.is_full_house())

    def possible_straight(self) -> bool:
        """Returns True if possible to draw to a straight."""
        return self.has_partial_straight() and not self._has_flush()

    def possible_3kind(self) -> bool:
        """Returns True if possible to draw to three of a kind."""
        return (self.num_ranks.max(initial=0) + self.num_empty_slots() >= 3 and
                not self._has_at_least_xkind(4) and
                not self._has_full_house() and
                not self._has_flush() and
                not self.num_pair() == 2)

    def possible_2pair(self) -> bool:
        """Returns True if possible to draw to two pair."""
        return (self.num_unique_ranks() <= 3 and
                not self._has_at_least_xkind(3) and
                not self.is_flush())

    def possible_jack_high_pair(self) -> bool:
        """Returns True if possible to draw to a pair of jacks or better."""
        max_num_jack_or_higher = self.num_ranks[[0, 10, 11, 12]].max(initial=0)
        return (max_num_jack_or_higher + self.num_empty_slots() >= 2 and
                not np.any(self.num_ranks[1:10] >= 2) and
                not self.num_pair() >= 2 and
                not self._has_at_least_xkind(3) and
                not self._has_flush() and
                not self._has_straight())
