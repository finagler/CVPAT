import numpy as np
from numpy import testing as npt

from agent import advantage_estimator_trainer
from game import card
from game import payout_table


def test_get_target_advantage():
    # Cards don't matter for this method.
    cards = card.card_list('AC AC AC AC AC AC AC AC AC AC '
                           'AC AC AC AC AC AC AC AC AC AC ')
    # Expected and actual rewards are just [0, 1, 2, 3, 4, 5] for
    # all 11 playable hand-start positions.
    r_expected = np.repeat(np.arange(6).reshape((1, 6)), 11, axis=0)
    r_actual = np.repeat(np.arange(6).reshape((1, 6)), 11, axis=0)
    rewards = advantage_estimator_trainer.ShoeRewards(
        cards, r_expected, r_actual)
    prev_advantage_est = np.zeros((1, 6))
    target, mean, v_actual = advantage_estimator_trainer.get_target_advantage(
        rewards=rewards, max_hand_limit=0, advantage_est=prev_advantage_est)
    target_means = target[0].mean(axis=-1)
    expected_means = np.zeros((11,))
    npt.assert_allclose(target_means, expected_means, atol=1e-10)
    expected_unnormed = np.array(
        [[[5, 5, 5, 5, 5, 10],
          [0, 5, 5, 5, 5, 5],
          [0, 0, 5, 5, 5, 5],
          [0, 0, 0, 5, 5, 5],
          [0, 0, 0, 0, 5, 5],
          [0, 0, 0, 0, 0, 5],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0]]]).astype(float)
    npt.assert_allclose(target + mean, expected_unnormed)


def test_generate_random_shoe_reward_arrays():
    data = advantage_estimator_trainer._generate_random_shoe_reward_arrays(
        num_shoes=2, rng=np.random.default_rng(seed=1),
        num_decks=2, payout=payout_table.PayoutTable.default())
    assert data.shape == (2, 104, 13)
