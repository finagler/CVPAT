"""Class representing a payout table for video poker.

Payout table for all possible poker hands. Ante (the initial bet, if any) is
implemented as an overall bias across all hands, including the 'no hand' type.
"""

import numpy as np


HANDS_TYPE_NAMES = ('royal', 's. flush', '5 kind', '4 kind', 'f.house',
                    'flush', 'straight', '3 kind', '2 pair', 'j+ pair',
                    'low pair', 'no hand')


class PayoutTable(object):
    """Table of payout for each type of winning poker hand."""
    def __init__(self, royal_flush: int, straight_flush: int, five_kind: int,
                 four_kind: int, full_house: int, flush: int, straight: int,
                 three_kind: int, two_pair: int, jack_high_pair: int,
                 low_pair: int, ante: int = 0):
        self.table = np.array([royal_flush, straight_flush, five_kind,
                               four_kind, full_house, flush, straight,
                               three_kind, two_pair, jack_high_pair,
                               low_pair, 0]
                              ).astype(int) - ante

    def __repr__(self):
        return (f'rf: {self.table[0]}, sf: {self.table[1]}, ' +
                f'5k: {self.table[2]}, 4k: {self.table[3]}, ' +
                f'fh: {self.table[4]}, fl: {self.table[5]}, ' +
                f'st: {self.table[6]}, 3k: {self.table[7]}, ' +
                f'2p: {self.table[8]}, j+: {self.table[9]}, ' +
                f'1p: {self.table[10]}, ante: {self.table[11]}')

    @classmethod
    def default(cls, ante: int = 0):
        return PayoutTable(
            10000, 1000, 200, 80, 70, 60, 50, 20, 15, 10, 0, ante=ante)

    @property
    def royal_flush(self):
        return self.table[0]

    @royal_flush.setter
    def royal_flush(self, value: int):
        self.table[0] = value

    @property
    def straight_flush(self):
        return self.table[1]

    @straight_flush.setter
    def straight_flush(self, value: int):
        self.table[1] = value

    @property
    def five_kind(self):
        return self.table[2]

    @five_kind.setter
    def five_kind(self, value: int):
        self.table[2] = value

    @property
    def four_kind(self):
        return self.table[3]

    @four_kind.setter
    def four_kind(self, value: int):
        self.table[3] = value

    @property
    def full_house(self):
        return self.table[4]

    @full_house.setter
    def full_house(self, value: int):
        self.table[4] = value

    @property
    def flush(self):
        return self.table[5]

    @flush.setter
    def flush(self, value: int):
        self.table[5] = value

    @property
    def straight(self):
        return self.table[6]

    @straight.setter
    def straight(self, value: int):
        self.table[6] = value

    @property
    def three_kind(self):
        return self.table[7]

    @three_kind.setter
    def three_kind(self, value: int):
        self.table[7] = value

    @property
    def two_pair(self):
        return self.table[8]

    @two_pair.setter
    def two_pair(self, value: int):
        self.table[8] = value

    @property
    def jack_high_pair(self):
        return self.table[9]

    @jack_high_pair.setter
    def jack_high_pair(self, value: int):
        self.table[9] = value

    @property
    def low_pair(self):
        return self.table[10]

    @low_pair.setter
    def low_pair(self, value: int):
        self.table[10] = value

    @property
    def ante(self):
        return self.table[11]

    @ante.setter
    def ante(self, value: int):
        self.table[11] = value
