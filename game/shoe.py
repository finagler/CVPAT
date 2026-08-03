import itertools
import typing

import numpy as np

from game import card


def shuffled_cards(num_decks: int,
                   rng: np.random.Generator) -> typing.List[card.Card]:
    """Returns shuffled list of cards.

    Args:
        num_decks: number of decks list comprises.
        rng: random number generator to use for shuffling.

    Returns: shuffled list of 52 * num_decks cards.
    """
    card_iter = itertools.product(range(1, 14), card.SuitEnum)
    cards = [card.Card(r, s) for r, s in card_iter] * num_decks
    rng.shuffle(cards)
    return cards


class Shoe(object):
    """A dealer's shoe containing one or more (possibly partial) decks of cards.

    A shoe is a stack of cards from one or more standard 52-card decks (Ace
    through King, four-suit), some of which may have already been played.

    Args:
        card_list: which cards are currently in the shoe.
        num_decks: number of 52-card decks to initialize. Ignored if card_list
            is specified.
        rng: random generator used to shuffle cards.
    """
    def __init__(self,
                 card_list: typing.Optional[typing.Union[
                     str, typing.Iterable[card.Card]]] = None,
                 num_decks: int = 1,
                 rng: typing.Optional[np.random.Generator] = None):
        self._num_decks = num_decks
        self._rng = rng or np.random.default_rng()
        if isinstance(card_list, str):
            self._cards = card.card_list(card_list)
        elif card_list is None:
            self._cards = shuffled_cards(self._num_decks, self._rng)
        else:
            self._cards = list(card_list)

    @property
    def cards(self):
        return list(self._cards)

    @property
    def num_decks(self):
        return self._num_decks

    def size(self):
        return len(self._cards)

    @classmethod
    def shoe_excluding_cards(cls, exclude: typing.Iterable[card.Card],
                             num_decks: int = 1):
        """Returns a HandStats with all cards except those specified.

        Args:
            exclude: list of cards to exclude.
            num_decks: number of 52-card decks to start with.

        Returns:
            Shoe with specified cars
        """
        cards = [card.Card(rank, suit) for rank, suit in
                 itertools.product(range(1, 14), card.SuitEnum)] * num_decks
        for c in exclude:
            cards.remove(c)
        return Shoe(card_list=cards, num_decks=num_decks)

    @classmethod
    def shoe_with_num_cards_remaining(cls, num_cards, num_decks: int = 1):
        cards = shuffled_cards(num_decks=num_decks, rng=np.random.default_rng())
        return Shoe(cards[:num_cards], num_decks=num_decks)

    def draw_cards(self, num_cards=5) -> typing.List[card.Card]:
        """Deals a hand of 5 cards, removing them from the deck.

        Returns:
            hand_delt: the hand of 5 cards delt
        """
        draw = self._cards[:num_cards]
        self._cards = self._cards[num_cards:]
        return draw


class InteractiveShoe(Shoe):
    def __init__(self,
                 card_list: typing.Optional[typing.Union[
                     str, typing.Iterable[card.Card]]] = None,
                 num_decks: int = 1):
        super().__init__(card_list, num_decks)

    def draw_cards(self, num_cards=5) -> typing.List[card.Card]:
        """Deals a hand of 5 cards, removing them from the deck.

        Returns:
            hand_delt: the hand of 5 cards delt
        """
        draw_str = input(f'Draw {num_cards} cards: ')
        draw = card.card_list(draw_str)
        for c in draw:
            try:
                self._cards.remove(c)
            except ValueError as ex:
                raise ValueError(f'{c} not in shoe')
        return draw
