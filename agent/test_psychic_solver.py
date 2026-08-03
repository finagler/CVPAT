from numpy import testing as npt

from agent import psychic_solver
from game import payout_table
from game import card
from game import hand


def test_solve_all_no_hand_limit():
    """Test solve_all where hurting the first hand makes second hand better."""
    cards = card.card_list(
        'JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S 6C 9C 2D 8H')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0)
    solver = psychic_solver.PsychicSolver(
        cards, hand_limit=0, payout=payout, best_expected_only=False)
    values, num_held, held = solver.solve_all()
    # 17-card shoe
    assert values.shape == (17,)
    assert num_held.shape == (17,)
    assert len(held) == 17

    # First hand is a full house, but you can do better by discarding the pair
    # (settling for three-of-a-kind) to line up for the royal flush.
    npt.assert_allclose(values, [1002., 1001., 1000., 1., 0., 0., 0., 1000.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    npt.assert_allclose(num_held, [3, 4, 5, 1, 0, 0, 0, 4,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert held[0] == [hand.Hand(),
                       hand.Hand(card.card_list('8C')),
                       hand.Hand(card.card_list('JC JD')),
                       hand.Hand(card.card_list('JC JD JH')),
                       hand.Hand(card.card_list('8C JC JD JH')),
                       hand.Hand(card.card_list('8C 8D JC JD JH'))]


def test_solve_all_no_hand_limit_best_expected():
    """Test solve_all with best_expected_only."""
    # Set up a busted inside straight flush. If you know the shoe order then
    # the best move is to discard the 8 and go for the normal straight, but
    # any policy that optimizes for expected value would choose the 2 when
    # discarding just one card. Under that constraint the best move is to hold
    # just the two for two pair.
    cards = card.card_list('2C 4S 5S 6S 8S 3C 3D 2D JC 7S')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0)
    # Solve with both normal and best_expected_only. We expect the universal
    # solver to discard the
    solver = psychic_solver.PsychicSolver(
        cards, hand_limit=0, payout=payout, best_expected_only=False)
    values, num_held, held = solver.solve_all()
    npt.assert_allclose(values, [4., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    npt.assert_allclose(num_held, [4., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert held[0] == [hand.Hand(),
                       hand.Hand(card.card_list('2C')),
                       hand.Hand(card.card_list('2C 4S')),
                       hand.Hand(card.card_list('2C 4S 5S')),
                       hand.Hand(card.card_list('2C 4S 5S 6S')),
                       hand.Hand(card.card_list('2C 4S 5S 6S 8S'))]

    solver_exp = psychic_solver.PsychicSolver(
        cards, hand_limit=0, payout=payout, best_expected_only=True)
    values_exp, num_held_exp, held_exp = solver_exp.solve_all()
    npt.assert_allclose(values_exp, [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    npt.assert_allclose(num_held_exp, [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert held_exp[0] == [hand.Hand(),
                           hand.Hand(card.card_list('2C')),
                           hand.Hand(card.card_list('2C 4S')),
                           hand.Hand(card.card_list('4S 5S 6S')),
                           hand.Hand(card.card_list('4S 5S 6S 8S')),
                           hand.Hand(card.card_list('2C 4S 5S 6S 8S'))]


def test_solve_all_hand_limit_ordered():
    """Test solve_all where hurting the first hand makes second hand better."""
    cards = card.card_list(
        'JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S 6C 9C 2D 8H')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0)
    solver = psychic_solver.PsychicSolver(cards, hand_limit=1, payout=payout)
    values, num_held, held = solver.solve_all()
    # 15-card shoe, so 11 possible hands with a 5-card sliding window.
    assert len(held) == 17

    # First hand is a full house, but you can do better by discarding the pair
    # (settling for three-of-a-kind) to line up for the royal flush.
    npt.assert_allclose(
        values, [[7., 1., 1., 1., 0., 0., 0., 1000., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0.]])
    npt.assert_allclose(
        num_held, [[5, 2, 1, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert held[0] == [hand.Hand(),
                       hand.Hand(card.card_list('8C')),
                       hand.Hand(card.card_list('JC JD')),
                       hand.Hand(card.card_list('JC JD JH')),
                       hand.Hand(card.card_list('8C JC JD JH')),
                       hand.Hand(card.card_list('8C 8D JC JD JH'))]


def test_solve():
    """Test solve with hurting first hand to draw better second hand."""
    cards = card.card_list(
        'JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S 6C 9C 2D 8H AH')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0)
    solver = psychic_solver.PsychicSolver(cards, hand_limit=0, payout=payout)
    payout, held = solver.solve()
    assert payout == 1002.
    assert held == [hand.Hand(card.card_list('JC JD JH')),
                    hand.Hand(card.card_list('AS JS QS KS')),
                    None]


def test_solve_allow_short_shoe():
    """Test solve with allow_short_shoe."""
    cards = card.card_list(
        'JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S AC 9C 2D 8H AH')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0)
    solver = psychic_solver.PsychicSolver(cards, hand_limit=0, payout=payout,
                                          allow_short_shoe=True)
    payout, held = solver.solve()
    assert payout == 1003.
    assert held == [hand.Hand(card.card_list('JC JD JH')),
                    hand.Hand(card.card_list('AS JS QS KS')),
                    hand.Hand(card.card_list('AC AH'))]


def test_solve_hand_limit():
    """Test solve with hurting first hand to draw better second hand."""
    cards = card.card_list(
        'JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S AC 9C 2D 8H KC KD KH 3C 4S')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0)
    solver = psychic_solver.PsychicSolver(cards, hand_limit=2, payout=payout)
    payout, held = solver.solve()
    assert payout == 1002.
    assert held == [hand.Hand(card.card_list('JC JD JH')),
                    hand.Hand(card.card_list('AS JS QS KS'))]


def test_solve_all_hand_limit():
    """Verifies that solve_all with hand limit computes value correctly."""
    # Cards are all the same so all hands are five-of-a-kind regardless of play,
    # so total value should be hand_limit * five-of-a-kind payoff.
    cards = card.card_list('AS') * 20
    payout = payout_table.PayoutTable(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
    solver = psychic_solver.PsychicSolver(cards, hand_limit=2, payout=payout)
    values, num_held, held = solver.solve_all()
    # 11 ones + 9 zeros, 6 twos + 5 ones + 9 zeros
    expected = [[1.] * 11 + [0.] * 9,
                [2.] * 6 + [1.] * 5 + [0.] * 9]
    npt.assert_allclose(values, expected)
