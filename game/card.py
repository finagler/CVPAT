"""Class representing a single playing card from a 52-card / 4-suit deck."""
from __future__ import annotations

import enum
import functools
import typing


class SuitEnum(enum.Enum):
    CLUB = 0
    DIAMOND = 1
    HEART = 2
    SPADE = 3


RANK_SYMBOLS = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}


STRING_TO_RANK = {s: r for r, s in RANK_SYMBOLS.items()}


STRING_TO_SUIT = {'C': SuitEnum.CLUB, 'D': SuitEnum.DIAMOND,
                  'H': SuitEnum.HEART, 'S': SuitEnum.SPADE}


SUIT_SYMBOLS = {SuitEnum.CLUB: '\N{BLACK CLUB SUIT}',
                SuitEnum.DIAMOND: '\N{WHITE DIAMOND SUIT}',
                SuitEnum.HEART: '\N{WHITE HEART SUIT}',
                SuitEnum.SPADE: '\N{BLACK SPADE SUIT}'}


def card_list(cards_str: str) -> typing.List[Card]:
    """Returns list of cards from space-separated string, e.g. '2C 3C'"""
    return [Card.from_str(c) for c in cards_str.split()]


@functools.total_ordering
class Card(object):
    def __init__(self, rank: int, suit: SuitEnum):
        if rank < 1 or rank > 13:
            raise ValueError('rank must be between 1 and 13')
        self.rank = int(rank)
        self.suit = suit

    def __repr__(self):
        return f'{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}'

    def __lt__(self, other: Card) -> bool:
        if self.rank == other.rank:
            return self.suit.value < other.suit.value
        return self.rank < other.rank

    def __eq__(self, other: Card) -> bool:
        return self.rank == other.rank and self.suit.value == other.suit.value

    def __hash__(self):
        return hash((self.rank, self.suit))

    @classmethod
    def from_str(cls, string: str) -> Card:
        string = string.upper()
        return Card(STRING_TO_RANK[string[:-1]], STRING_TO_SUIT[string[-1]])

    @property
    def rank_index(self):
        return self.rank - 1

    @property
    def suit_index(self):
        return self.suit.value
