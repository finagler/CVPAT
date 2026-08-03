"""Controller for a game of one-shoe poker."""
from __future__ import annotations

import copy
import collections
import typing

import numpy as np

from game import hand
from game import shoe
from game import payout_table


GameMove = collections.namedtuple(
    'GameMove', ('initial_hand', 'held', 'final_hand', 'payout', 'info'))


class PokerGame(object):
    """A simple game of one-shoe poker as a human would play it.

    Attributes:
        num_decks: number of decks in starting shoe
        payout: payout table
        hand_limit: max number of hands, or zero for unlimited.
        draw_shoe: starting shoe, or None to create random shoe.
        allow_short_shoe: if True then allow play to end of the shoe,
            limiting replacements to remaining cards. If False then game
            ends when shoe has fewer than 10 cards before drawing new hand.
        verbose: if True then print drawn cards and payout.
        hand_number: number of hands played, including the current hand.
        score: cumulative payout so far.
        game_over: set to True if player has quit.
        history: list of moves conducted so far.
        hands_remaining: number of hands that can be played after current
            hand. If hand_limit is zero (unlimited) then this is the max number
            of hands before shoe is depleted with most conservative play (i.e.
            holding all cards). Otherwise this is equal to hand_limit minus
            hand_number.
    """
    def __init__(self, num_decks: typing.Optional[int] = None,
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 hand_limit: int = 0,
                 draw_shoe: typing.Optional[shoe.Shoe] = None,
                 allow_short_shoe: bool = False,
                 verbose: bool = False,
                 seed: typing.Optional[np.random.SeedSequence] = None):
        """Initialize game

        Args:
            num_decks: number of decks in starting shoe
            payout: payout table
            hand_limit: max number of hands, or zero for unlimited.
            draw_shoe: starting shoe, or None to create random shoe.
            allow_short_shoe: if True then allow play to end of the shoe,
                limiting replacements to remaining cards. If False then game
                ends when shoe has fewer than 10 cards before drawing new hand.
            verbose: if True then print drawn cards and payout.
            seed: random seed to use for shuffling cards.
        """
        if (num_decks is not None and draw_shoe is not None and
                draw_shoe.num_decks != num_decks):
            raise ValueError(
                f'number decks specified {num_decks} != shoe '
                f'{draw_shoe.num_decks}')
        self.payout = payout or payout_table.PayoutTable.default()
        self.hand_limit = hand_limit
        if seed is None:
            self.seed = np.random.SeedSequence()
        else:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.draw_shoe = draw_shoe or shoe.Shoe(num_decks=num_decks,
                                                rng=self.rng)
        self.allow_short_shoe = allow_short_shoe
        self.verbose = verbose
        self.hand = hand.Hand()
        self.hand_number = 0
        self.score: int = 0
        self._history: typing.Deque[GameMove] = collections.deque()
        self.game_over: bool = False

    @property
    def num_decks(self):
        return self.draw_shoe.num_decks

    @property
    def history(self) -> typing.List[GameMove]:
        return list(self._history)

    @property
    def hands_remaining(self):
        """Number of hands that can be played after the current hand.

        If hand_limit is 0 (unlimited) then this will return the max number of
        possible hands with the most conservative play (holding all cards).
        """
        if self.hand_limit:
            return self.hand_limit - self.hand_number
        if self.allow_short_shoe:
            return self.draw_shoe.size() // 5
        return (self.draw_shoe.size() - 9) // 5

    def copy(self) -> PokerGame:
        return copy.deepcopy(self)

    def reset(self):
        """Refills and shuffles shoe and deals new hand."""
        self.draw_shoe = shoe.Shoe(
            num_decks=self.num_decks, rng=self.rng)
        self.hand = hand.Hand()
        self.score = 0
        self._history.clear()
        self.hand_number = 0
        self.draw_new_hand()

    def _draw_to_fill_hand(self, held: hand.Hand) -> hand.Hand:
        """Draw from deck to fill hand to five cards."""
        cards = self.draw_shoe.draw_cards(held.num_empty_slots())
        assert set(held.cards).issubset(set(self.hand.cards))
        self.hand = held.add(cards)
        return self.hand

    def _payout(self) -> int:
        """Returns payout for current hand."""
        return (self.hand.eval_hand() * self.payout.table).sum()

    def shoe_depleted(self) -> bool:
        """Returns True if shoe does not have enough cards for another hand."""
        if self.allow_short_shoe:
            return self.draw_shoe.size() < 5
        return self.draw_shoe.size() < 10

    def draw_new_hand(self) -> bool:
        """Discards hand and draws a new one.

        Returns:
            depleted: True if shoe is depleted before drawing new hand.
        """
        if (self.game_over or self.shoe_depleted() or
                (self.hand_limit and self.hand_number >= self.hand_limit)):
            self.hand = hand.Hand()
            if self.verbose:
                print(f'Game over, final score {self.score}')
            return True
        self.hand = hand.Hand(self.draw_shoe.draw_cards(5))
        self.hand_number += 1
        if self.verbose:
            print(f'New hand:  {self.hand}')
        return False

    def discard_and_replace(self, held: hand.Hand,
                            info: typing.Optional = None) -> float:
        """Discard and replace any cards not in 'held', and collect winnings.

        Args:
            held: cards that are to be held. Assumed to be a subset of current
                hand, though this isn't checked.
            info: additional analysis info to be included in history.

        Returns:
            any winnings from hand after replacement, given payout table.
        """
        initial_hand = self.hand
        final_hand = self._draw_to_fill_hand(held)
        if self.verbose:
            print(f'    final hand: {final_hand}')
        payout = int(self._payout())
        self.score += payout
        self._history.append(
            GameMove(initial_hand, held, final_hand, payout, info))
        if self.verbose and payout:
            print(f'    payout: {payout}, total: {self.score}')
        return payout
