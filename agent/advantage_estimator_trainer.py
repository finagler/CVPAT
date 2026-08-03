"""Class for evaluating all possible actions and positions in a shoe."""
from __future__ import annotations

import collections
import functools
import multiprocessing
import pathlib
import time
import typing
import warnings

import numpy as np
# import torch
# from torch import optim
# from torch.utils import tensorboard

from agent import advantage_estimator
from game import card
from game import hand
from game import payout_table
from stats import hand_stats
from game import shoe

# from ml import torch_util


class _RunningMeanAndVar(object):
    """Computes running mean and variance.

    Simple online computation of mean and variance using Welford's algorithm.

    Attributes:
        count (int): The number of accumulated samples.
        mean (array(D,)): Mean of the accumulated samples.
        var_s (array(D,)): Sample variance of the accumulated samples.
        var_p (array(D,)): Population variance of the accumulated samples.
    """
    def __init__(self):
        self.__shape = None
        # current attribute values
        self.__count = 0
        self.__m = None
        self.__s = None

    @property
    def count(self):
        return self.__count

    @property
    def mean(self):
        return self.__m

    @property
    def var_s(self):
        return self.__getvars(ddof=1)

    @property
    def var_p(self):
        return self.__getvars(ddof=0)

    def update(self, element):
        """Add one data sample.

        Args:
            element (array(D, )): data sample.
        """
        # Initialize if not yet.
        if self.__shape is None:
            self.__shape = element.shape
            self.__m = np.zeros(element.shape)
            self.__s = np.zeros(element.shape)
            self.__init_old_with_nan()
        # argument check if already initialized
        else:
            assert element.shape == self.__shape

        # Welford's algorithm
        self.__count += 1
        delta = element - self.__m
        self.__m += delta / self.__count
        self.__s += delta * (element - self.__m)

    def add_all(self, elements, backup_flg=True):
        """ add_all

        add multiple data samples.

        Args:
            elements (array(S, D)): data samples.
            backup_flg (boolean): if True, backup previous state for rollbacking.

        """
        # backup for rollbacking
        if backup_flg:
            self.__backup_attrs()

        for elem in elements:
            self.add(elem, backup_flg=False)

    def __getvars(self, ddof):
        if self.__count <= 0:
            return None
        min_count = ddof
        if self.__count <= min_count:
            return np.full(self.__shape, np.nan)
        else:
            return self.__s / (self.__count - ddof)



def _expected_payouts_by_num_held(
        cards: typing.List[card.Card],
        payout: typing.Optional[payout_table.PayoutTable] = None
        ) -> typing.Tuple[np.ndarray, typing.List[typing.List[hand.Hand]]]:
    """Returns expected payout at each point in shoe for number held.

    Returns an array where each element is the expected payout (i.e. if
    you know shoe contents but not order) for a single hand starting at a
    given point in the shoe and holding a given number of cards (discarding
    and replacing the rest).

    Returns:
        expected: ndarray of shape (shoe_size - 10, 6), where expected[i, h]
            is the expected payout (given remaining shoe contents but not
            order) of the 5-card hand from shoe.cards[i:i+5] after
            discarding and replacing (5-h) cards.
        held: 2d list of Hands, where held[i][h] is the hand that should
            be held when discarding (5-h) cards.
    """
    last_index = len(cards) - 10
    expected = collections.deque()
    held = collections.deque()
    for i in range(last_index + 1):
        delt = hand.Hand(cards[i:i+5])
        hs = hand_stats.HandStats(
            cards=cards[i+5:], payout=payout)
        exp_payout = hs.best_expected_payout_per_num_held(delt)
        expected.append([b.value for b in exp_payout])
        held.append([b.held for b in exp_payout])
    return np.array(expected), list(held)


def _actual_payouts_by_num_held(
        cards: typing.List[card.Card],
        held: typing.List[typing.List[hand.Hand]],
        payout: typing.Optional[payout_table.PayoutTable] = None,
        ) -> np.ndarray:
    """Returns actual payout at each point in shoe for number held.

    Returns an array where each element is the actual payout for a single
    hand starting at a given point in the shoe and holding a given number
    of cards (discarding and replacing the rest).

    Args:
        cards: list of cards in the shoe.
        held: 2d list of Hands, where held[i][h] is the hand that should
            be held when discarding (5-h) cards (e.g. as returned by
            _expected_payouts_by_num_held).
        payout: payout table to use.

    Returns:
        actual: ndarray of shape (shoe_size - 10, 6), where actual[i, h]
            is the actual payout of the 5-card hand from shoe.cards[i:i+5]
            after discarding and replacing the (5-h) cards specified in
            held.
    """
    last_index = len(cards) - 10
    payouts = np.zeros((last_index + 1, 6))
    for p in range(last_index + 1):
        replacements = cards[p+5:p+10]
        for h in range(6):
            filled = held[p][h].add(replacements[:5-h])
            payouts[p, h] = filled.payout(payout)
    return payouts


class ShoeRewards(object):
    """Expected and actual rewards for each num held cards and position."""
    def __init__(self, cards: typing.List[card.Card],
                 r_expected: np.ndarray, r_actual: np.ndarray):
        self.cards = cards
        self.r_expected = r_expected
        self.r_actual = r_actual

    @classmethod
    def from_array(cls, data: np.ndarray):
        """Returns ShoeRewards created from array."""
        return ShoeRewards(
            cards=[card.Card(n % 13 + 1, card.SuitEnum(n // 13))
                   for n in data[:, 0]],
            r_expected=data[:-9, 1:7], r_actual=data[:-9, 7:])

    def as_array(self) -> np.ndarray:
        """Returns data as 1d array of values in range [0, 51]."""
        data = np.zeros((len(self.cards), 13), dtype=np.float16)
        data[:, 0] = [c.suit_index * 13 + c.rank_index for c in self.cards]
        data[:-9, 1:7] = self.r_expected
        data[:-9, 7:] = self.r_actual
        return data


def _compute_shoe_rewards(
        cards: typing.List[card.Card],
        payout: payout_table.PayoutTable) -> ShoeRewards:
    r_expected, held = _expected_payouts_by_num_held(cards, payout)
    r_actual = _actual_payouts_by_num_held(cards, held, payout)
    return ShoeRewards(cards, r_expected, r_actual)


def _generate_random_shoe_reward_arrays(
        num_shoes: int,
        rng: np.random.Generator,
        num_decks: int = 4,
        payout: typing.Optional[payout_table.PayoutTable] = None) -> np.ndarray:
    """Returns random shoes along with expected and actual rewards at each hand.

    Returns a array of shape (num_shoes, 52 * num_decks, 13) representing the
    expected and actual rewards for each possible number of cards held (0-5)
    and starting hand position in the shoe. Specifically:

    data[s, p, 0]: card at position <p> in shoe <s>, encoded as a float16.
    data[s, p, h = 1:7]: max expected reward (accounting for remaining shoe
        contents but not order) when playing shoe <s>, hand starting at
        position <p>, and holding <h>-1 cards.
    data[s, p, h = 7:]: actual reward when playing shoe <s>, hand starting at
        position <p>, and holding <h>-7 cards.

    Args:
        num_shoes: number of shoes to generate
        rng: random generator to use to shuffle cards
        num_decks: number of decks in each shoe
        payout: payout table to use

    Returns: data, and ndarray of shape (num_shoes, 52 * num_decks, 13), with
        dtype np.float16.
    """
    data_q = collections.deque()
    cards = shoe.shuffled_cards(num_decks=num_decks, rng=rng)
    for _ in range(num_shoes):
        rng.shuffle(cards)
        data_q.append(_compute_shoe_rewards(cards, payout).as_array())
    return np.array(data_q).astype(np.float16)


def get_target_advantage(rewards: ShoeRewards,
                         max_hand_limit: int,
                         advantage_est: np.ndarray
                         ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns target advantages for shoe.

    Args:
        rewards: expected and actual reward for hands starting at each point
            in a shoe.
        max_hand_limit: max hand limit (including first hand), or zero for
            no hand limit.
        advantage_est: previous estimated advantage for action at position, i.e.
             A_e(λ, p, a | θ), where:
                 λ = remaining hand limit after current hand
                 p = index of first card of current hand in shoe
                 a = number of cards to hold (action)
             Shape: (max_hand_limit, playable_shoe_size, 6)
             If max_hand_limit is zero then shape will be
             (1, playable_shoe_size, 6) and the first index is ignored.

    Returns:
        advantage_target: mean-normalized target advantages for hand-limit,
            position and num_held. Shape: (hand_limit, playable_shoe_size, 6)
        advantage_target_mean: mean of advantage_target before normalization.
        v_actual: actual value of each hand.
            Shape: (max_hand_limit, playable_shoe_size)
    """
    # A max_hand_limit of zero indicates no limit.
    _unlimited = (max_hand_limit == 0)
    if _unlimited:
        hand_limit_dim = 1
    else:
        hand_limit_dim = max_hand_limit + 1

    # Using the current advantage estimator, compute how many cards will be
    # held for every possible hand starting position and hand-limit in
    # the shoe.
    cards = rewards.cards
    # Number of cards in the shoe.
    shoe_size = len(cards)
    # The window of playable hand positions in the shoe.
    playable_shoe_size = shoe_size - 10
    # Expected reward and num cards held: R_exp(p, a) = E[R(p, a)], H(p, a)
    # Shape: (playable_shoe_size, 6)
    r_expected = rewards.r_expected[:playable_shoe_size+1, :]
    # Actual reward for each position and num cards held: R(p, a)
    # Shape: (playable_shoe_size, 6)
    r_actual = rewards.r_actual[:playable_shoe_size+1, :]
    # Estimated Q value for current hand_limit:
    # Q_e(λ, p, a | θ) = R_exp(p, a) + A_e(λ, p, a | θ)
    # Shape: (1, playable_shoe_size, 6)
    q_est = r_expected[np.newaxis, :, :] + advantage_est
    # Number of cards held in each state (i.e. the action taken) using the
    # current advantage estimator.
    # H(λ, p | θ) = argmax_a[Q_e(λ, p, a | θ)]
    # Shape: (1, playable_shoe_size)
    num_held = np.argmax(q_est, axis=-1)

    # Now compute the value and Q-value for every hand-limit,
    # position and num_held (action) given current params θ, iterating
    # backwards through the shoe with increasing hand limits.
    last_index = playable_shoe_size
    # V(λ, p | θ): Cumulative reward starting at (λ, p) and following
    #     policy given by θ.
    # Shape: (max_hand_limit, shoe_size + 1)
    v_actual = np.zeros((hand_limit_dim, shoe_size + 1)).astype(float)
    # A(λ, p | θ): Expected advantage.
    # Shape: (max_hand_limit, playable_shoe_size, 6)
    advantage_target = np.zeros((hand_limit_dim, playable_shoe_size + 1, 6))
    if _unlimited:
        for p in range(last_index, -1, -1):
            # Q(p, a | θ) = R(p, a) + V(p + 10 - a | θ)
            #     actual cumulative reward when choosing a at p and
            #     following the policy given by θ afterwards.
            # Shape: (6,)
            q_actual = (r_actual[p, :] + v_actual[0, p + 10:p + 4:-1])
            # V(p | θ) = Q(p, H(λ, p | θ))
            v_actual[0, p] = q_actual[num_held[0, p]]
            # A(p | θ) ← Q(p, a | θ) - R_exp(p, a)
            # This works because:
            #     E[A(p | θ)] = E[Q(p, a | θ) - R(p, a)]
            #                    = E[Q(p, a | θ)] - R_exp(p, a)
            # We know R_exp exactly, but will substitute Q for E[Q] and
            # rely on sampling to drive our estimate closer to the true
            # estimate.
            advantage_target[0, p, :] = q_actual - r_expected[p, :]
    else:
        # Same as unlimited, except we iterate through hand limits and compute
        # q_actual using v_actual for the previous column.
        for hand_limit in range(1, hand_limit_dim):
            for p in range(last_index, -1, -1):
                # Q(λ, p, a | θ) = R(p, a) + V(λ-1, p + 10 - a | θ)
                q_actual = (r_actual[p, :] +
                            v_actual[hand_limit-1, p + 10:p + 4:-1])
                # V(λ, p | θ) = Q(λ, p, H(λ, p | θ))
                v_actual[hand_limit, p] = q_actual[num_held[hand_limit, p]]
                # A(λ, p | θ) ← Q(λ, p, a | θ) - R_exp(p, a)
                advantage_target[hand_limit, p, :] = q_actual - r_expected[p, :]

    # Zero-mean the target advantage.
    advantage_target_mean = advantage_target.mean(axis=-1, keepdims=True)
    advantage_target -= advantage_target_mean
    v_actual = v_actual[1:, :playable_shoe_size]
    return advantage_target, advantage_target_mean, v_actual


class TableAdvantageEstimatorTrainer(object):
    def __init__(self,
                 # Basic configuration
                 data_dir: str,
                 rng: typing.Optional[np.random.Generator] = None,
                 exp_name: str = 'PokerAdvantage',
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 num_decks: int = 4,
                 checkpoint_dir: typing.Optional[str] = None,
                 run_name: typing.Optional[str] = None,
                 checkpoint_interval: int = 10,
                 max_shoes_per_epoch: typing.Optional[int] = None):
        """Initialization."""
        self.data_dir = pathlib.Path(data_dir)
        self.rng = rng or np.random.default_rng()
        self.exp_name = exp_name
        self.payout = payout
        self.num_decks = num_decks
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.checkpoint_interval = checkpoint_interval
        self.max_shoes_per_epoch = max_shoes_per_epoch

        self._max_hand_limit = self.num_decks * 52 // 5 - 1
        self._playable_shoe_size = self.num_decks * 52 - 9
        self.advantage = np.zeros(
            (self._max_hand_limit+1, self._playable_shoe_size, 6))
        self.rewards = None  # loaded in train_one_epoch

    def load_advantage(self, path: str):
        self.advantage = np.load(path)

    def _load_reward_data(self):
        files = [f for f in self.data_dir.glob('*.npy')]
        if self.max_shoes_per_epoch:
            self.rng.shuffle(files)
            files = files[:self.max_shoes_per_epoch // 500]
        shoes_q = collections.deque()
        for f in files:
            shoes_q.append(np.load(str(f)))
        self.rewards = np.vstack(shoes_q)

    def train_one_epoch(self, num_shoes: typing.Optional[int] = None):
        self._load_reward_data()
        num_shoes = num_shoes or self.rewards.shape[0]
        target_advantage_mean = torch_util.RunningMeanVar()
        v_actual_mean = torch_util.RunningMeanVar()
        for shoe_num in range(num_shoes):
            if shoe_num % 10000 == 0:
                print(f'training {shoe_num}/{num_shoes}')
            shoe_rewards = ShoeRewards.from_array(self.rewards[shoe_num])
            target_advantage, _, v_actual = get_target_advantage(
                shoe_rewards, self._max_hand_limit, self.advantage)
            target_advantage_mean.update(target_advantage)
            v_actual_mean.update(v_actual)
        return (target_advantage_mean.mean,
                target_advantage_mean.sample_variance,
                v_actual_mean.mean, v_actual_mean.variance)

    def train(self, lr: float, num_epochs: int = 5,
              num_shoes: typing.Optional[int] = None):
        target_adv = target_adv_var = None
        v_actual_mean = v_actual_var = None
        for epoch in range(num_epochs):
            target_adv, target_adv_var, v_actual_mean, v_actual_var = (
                self.train_one_epoch(num_shoes))
            self.advantage = (1 - lr) * self.advantage + lr * target_adv
            if epoch % self.checkpoint_interval == 0:
                fname = f'advantage-mean-{self.num_decks}deck-{epoch:05}.npy'
                path = f'{self.checkpoint_dir}/{fname}'
                print(f'saving to {path}')
                np.save(path, self.advantage)
        return target_adv, target_adv_var, v_actual_mean, v_actual_var

    # def train_one_epoch(self, num_shoes: typing.Optional[int] = None):
    #     num_shoes = num_shoes or self.rewards.shape[0]
    #     self._accumulator = np.zeros(
    #         (num_shoes, self._max_hand_limit+1, self._playable_shoe_size, 6))
    #     for shoe_num in range(num_shoes):
    #         if shoe_num % 10000 == 0:
    #             print(f'training {shoe_num}/{num_shoes}')
    #         shoe_rewards = ShoeRewards.from_array(self.rewards[shoe_num])
    #         target_advantage, _ = get_target_advantage(
    #             shoe_rewards, self._max_hand_limit, self.advantage)
    #         self._accumulator[shoe_num, :, :, :] = target_advantage
    #     # np.save(
    #     #     '/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints/advantage-full-2deck-2.npy',
    #     #     self._accumulator)
    #     return self._accumulator.mean(axis=0), self._accumulator.var(axis=0)


class TargetAdvantageBuffer:
    """Buffer for holding state -> target advantage pairs.

    Attributes:
        mini_batch_size: size of each sample.
    """
    def __init__(self,
                 size: int,
                 mini_batch_size: int,
                 rng: np.random.Generator,
                 state_dim: int = advantage_estimator.STATE_DIM,
                 target_dim: int = 6):
        self.mini_batch_size = mini_batch_size
        self._state_buf = np.full(
            (size, state_dim), np.nan, dtype=advantage_estimator.STATE_DTYPE)
        self._target_buf = np.full(
            (size, target_dim), np.nan, dtype=advantage_estimator.STATE_DTYPE)
        self._next = 0
        self._rng = rng

    @property
    def capacity(self):
        return self._state_buf.shape[0]

    @property
    def size(self):
        return self._next

    @property
    def num_empty_slots(self):
        return max(0, self.capacity - self.size)

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator):
        self._rng = rng

    def clear(self):
        self._next = 0

    def store(self, states: np.ndarray, targets: np.ndarray,
              truncate: bool = False) -> bool:
        states = states.reshape(-1, advantage_estimator.STATE_DIM)
        targets = targets.reshape(-1, 6)
        assert states.shape[0] == targets.shape[0]
        num_entries = states.shape[0]
        if self.num_empty_slots < num_entries:
            msg = (f'Buffer overflow, truncating storage of {num_entries} ' +
                   f'entries to {self.num_empty_slots}')
            if truncate:
                warnings.warn(msg)
                states = states[:self.num_empty_slots]
                targets = targets[:self.num_empty_slots]
                num_entries = states.shape[0]
            else:
                raise ValueError(msg)
        self._state_buf[self._next:self._next + num_entries] = states
        self._target_buf[self._next:self._next + num_entries] = targets
        self._next += num_entries
        return self.num_empty_slots > 0

    def sample(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
        num_batches = self.size // self.mini_batch_size
        batch_indexes = self._rng.choice(
            self.size, (num_batches, self.mini_batch_size), replace=False)
        for indexes in batch_indexes:
            states = self._state_buf[indexes]
            targets = self._target_buf[indexes]
            yield states, targets

    def __len__(self) -> int:
        return self.size


class AdvantageEstimatorTrainer(object):
    """Trainer for NNAdvantageEstimator."""
    def __init__(self,
                 # Basic configuration
                 exp_name: str = 'PokerAdvantage',
                 payout: typing.Optional[payout_table.PayoutTable] = None,
                 num_decks: int = 4,
                 log_dir: typing.Optional[str] = None,
                 checkpoint_dir: typing.Optional[str] = None,
                 run_name: typing.Optional[str] = None,
                 checkpoint_interval: int = 10,
                 target_copy_interval: int = 10,
                 unlimited: bool = True,
                 # Hyperparams
                 shoes_per_batch: int = 2000,
                 mini_batch_size: int = 52,
                 device: str = 'auto',
                 learning_rate: float = 0.1,
                 seed: typing.Optional[int] = None,
                 # Status and checkpoint
                 epoch: int = 0,
                 update_num: int = 0,
                 rng: typing.Optional[np.random.Generator] = None,
                 estimator_state_dict: typing.Optional[typing.Dict] = None,
                 optimizer_state_dict: typing.Optional[typing.Dict] = None):
        """Initialization."""
        # Basic config
        self.exp_name = exp_name
        self.payout = payout
        self.num_decks = num_decks
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.checkpoint_interval = checkpoint_interval
        self.target_copy_interval = target_copy_interval
        self.unlimited = unlimited

        # Hyperparams
        self.shoes_per_batch = shoes_per_batch
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.learning_rate = learning_rate
        if seed is None:
            self.seed = np.random.default_rng().integers(
                low=0, high=np.iinfo(np.int64).max)
        else:
            self.seed = seed

        # Trainer state (from checkpoints)
        self.epoch = epoch
        self.update_num = update_num
        if rng:
            self._rng = rng
        else:
            self._rng = np.random.default_rng(seed=self.seed)
        self._device = torch_util.torch_device(self.device)
        self._estimator = advantage_estimator.NNAdvantageEstimator(
            self._device, unlimited=self.unlimited)
        if estimator_state_dict:
            self._estimator.load_state_dict(estimator_state_dict)
        self._target_estimator = advantage_estimator.NNAdvantageEstimator(
            self._device, unlimited=self.unlimited)
        self._target_estimator.load_state_dict(
            self._estimator.anet.state_dict())
        self._optimizer = optim.Adam(self._target_estimator.anet.parameters(),
                                     lr=self.learning_rate, eps=1e-5)
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)

        # Derived values and data structures
        self._max_hand_limit = self.num_decks * 52 // 5 - 1
        self._batch_size = (self.shoes_per_batch * (self.num_decks * 52 - 9) *
                            (self._max_hand_limit + 1))
        if self._batch_size % self.mini_batch_size:
            ValueError(f'batch size ({self._batch_size}) must be multiple of ' +
                       f'mini-batch size ({mini_batch_size})')
        self._buffer = TargetAdvantageBuffer(
            self._batch_size, self.mini_batch_size, rng=self.rng)
        if not self.run_name:
            self.run_name = f'{self.exp_name}__{self.seed}__{int(time.time())}'
        self.writer = None
        if self.log_dir:
            log_path = pathlib.Path(self.log_dir) / self.run_name
            if log_path.exists():
                self.writer = tensorboard.SummaryWriter(str(log_path))
            else:
                self.writer = tensorboard.SummaryWriter(str(log_path))
                param_strs = [f'|{k}|{v}|' for k, v in self.params.items()]
                self.writer.add_text(
                    'hyperparameters',
                    '|param|value|\n|-|-|\n' + '\n'.join(param_strs))

    @property
    def params(self):
        return dict(exp_name=self.exp_name,
                    payout=self.payout,
                    num_decks=self.num_decks,
                    log_dir=self.log_dir,
                    checkpoint_dir=self.checkpoint_dir,
                    run_name=self.run_name,
                    checkpoint_interval=self.checkpoint_interval,
                    target_copy_interval=self.target_copy_interval,
                    unlimited=self.unlimited,
                    shoes_per_batch=self.shoes_per_batch,
                    mini_batch_size=self.mini_batch_size,
                    device=self._device,
                    learning_rate=self.learning_rate,
                    seed=self.seed)

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator):
        self._rng = rng
        self._buffer.rng = rng

    @property
    def state(self):
        return dict(epoch=self.epoch,
                    update_num=self.update_num,
                    rng=self.rng,
                    estimator_state_dict=self._estimator.anet.state_dict(),
                    optimizer_state_dict=self._optimizer.state_dict())

    @classmethod
    def load(cls, path: str, **kwargs):
        params = torch.load(path)
        params.update(kwargs)
        return AdvantageEstimatorTrainer(**params)

    def save(self, path: str):
        full_params = self.params
        full_params.update(self.state)
        torch.save(full_params, path)

    def _refill_buffer(self, path: typing.Optional[pathlib.Path] = None):
        if path is None:
            data = _generate_random_shoe_reward_arrays(
                num_shoes=1, rng=self.rng, num_decks=self.num_decks,
                payout=self.payout)
        else:
            data = np.load(str(path))
        num_shoes = data.shape[0]
        for shoe_num in range(num_shoes):
            rewards: ShoeRewards = ShoeRewards.from_array(data[shoe_num])
            # State vectors for computing advantage: S(λ, p)
            # Shape: (hand_limit, playable_shoe_size, STATE_DIM)
            states = advantage_estimator.preprocess_state_vector(
                advantage_estimator.shoe_to_unnormed_states(
                    rewards.cards, self._max_hand_limit))
            with torch.no_grad():
                # Estimated advantage for action at position: A_e(λ, p, a | θ)
                # Shape: (hand_limit, playable_shoe_size, 6)
                advantage_est = self._estimator.get_advantage_for_states_numpy(
                    states)
            advantage_target, _, _ = get_target_advantage(
                rewards, max_hand_limit=self._max_hand_limit,
                advantage_est=advantage_est)
            self._buffer.store(states, advantage_target)

    def update_model(self, states, targets) -> float:
        """Update the model by gradient descent."""
        estimated_advantage = self._estimator.get_advantage_for_states(states)
        targets_t = torch.FloatTensor(targets).to(self._device)
        loss = torch.square(estimated_advantage - targets_t).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        max_estimate = estimated_advantage.abs().max().item()
        max_target = targets_t.abs().max().item()
        self._log_update(loss=loss, max_estimate=max_estimate,
                         max_target=max_target)
        return loss.item()

    def _log_update(self, loss: float, max_estimate: float, max_target: float):
        if self.writer:
            self.writer.add_scalar(
                'charts/loss_per_update', loss, self.update_num)
            self.writer.add_scalar(
                'charts/max_estimate_per_update', max_estimate, self.update_num)
            self.writer.add_scalar(
                'charts/max_target_per_update', max_target, self.update_num)

    def _log_epoch(self, loss: float, seconds_per_store: float,
                   seconds_per_train: float):
        if self.writer:
            self.writer.add_scalar(
                'charts/mean_loss_epoch', loss, self.epoch)
            self.writer.add_scalar(
                    'charts/seconds_per_epoch_store', seconds_per_store,
                    self.epoch)
            self.writer.add_scalar(
                'charts/seconds_per_epoch_train', seconds_per_train, self.epoch)

    def _maybe_save_checkpoint(self):
        if self.checkpoint_dir and self.epoch % self.checkpoint_interval == 0:
            out_fname = pathlib.Path(f'{self.run_name}-{self.epoch:04d}.zip')
            out_path = pathlib.Path(self.checkpoint_dir) / out_fname
            self.save(str(out_path))

    def _maybe_copy_target_to_base(self):
        if self.epoch % self.target_copy_interval == 0:
            state_dict = self._target_estimator.anet.state_dict()
            self._estimator.load_state_dict(state_dict)

    def train_from_files(self, num_epochs: int, data_dir: typing.Optional[str]):
        files = []
        while self.epoch < num_epochs:
            self._maybe_copy_target_to_base()
            self._maybe_save_checkpoint()
            epoch_start = time.time()
            losses = collections.deque()
            if not files:
                data_path = pathlib.Path(data_dir)
                files = [f for f in data_path.glob('*.npy')]
                self.rng.shuffle(files)
            if not files:
                raise FileNotFoundError(f'No data files found in {data_dir}')

            self._buffer.clear()
            while self._buffer.size < self._batch_size:
                data_file = files.pop()
                print(f'Loading {data_file} for epoch {self.epoch}')
                self._refill_buffer(data_file)
            seconds_per_store = time.time() - epoch_start
            print(f'storing for epoch {self.epoch} took ',
                  f'{seconds_per_store} seconds, buffered ',
                  f'{self._buffer.size} rows')

            for states, targets in self._buffer.sample():
                loss = self.update_model(states, targets)
                losses.append(loss)
                self.update_num += 1
            seconds_per_train = time.time() - epoch_start - seconds_per_store
            self._log_epoch(np.array(losses).mean(),
                            seconds_per_store=seconds_per_store,
                            seconds_per_train=seconds_per_train)
            self.epoch += 1

    def get_mean_value(self, data_dir: str):
        value_mean = torch_util.RunningMeanVar()
        data_path = pathlib.Path(data_dir)
        paths = [p for p in data_path.glob('*.npy')]
        for pnum, path in enumerate(paths):
            print(f'{pnum}/{len(paths)}: {path}')
            data = np.load(str(path))
            num_shoes = data.shape[0]
            for shoe_num in range(num_shoes):
                rewards: ShoeRewards = ShoeRewards.from_array(data[shoe_num])
                # State vectors for computing advantage: S(λ, p)
                # Shape: (hand_limit, playable_shoe_size, 53)
                states = advantage_estimator.preprocess_state_vector(
                    advantage_estimator.shoe_to_unnormed_states(
                        rewards.cards, self._max_hand_limit))
                with torch.no_grad():
                    # Estimated advantage at position: A_e(λ, p, a | θ)
                    # Shape: (hand_limit, playable_shoe_size, 6)
                    adv_est = self._estimator.get_advantage_for_states_numpy(
                        states)
                _, _, value = get_target_advantage(
                    rewards, max_hand_limit=self._max_hand_limit,
                    advantage_est=adv_est)
                value_mean.update(value)
        return value_mean.mean, value_mean.variance


def cache_shoe_reward_arrays(
        out_dir: str,
        shoes_per_file: int,
        prefix: str = 'rewards',
        num_decks: int = 3,
        max_num_files: int = 1000,
        start_num: int = 0,
        seed: typing.Optional[int] = None,
        payout: typing.Optional[payout_table.PayoutTable] = None):
    payout = payout or payout_table.PayoutTable.default()
    rng = np.random.default_rng(seed=seed)
    for fnum in range(start_num, max_num_files):
        fname = f'{prefix}-s{seed}-nd{num_decks}-{fnum:04d}.npy'
        out_path = pathlib.Path(out_dir) / fname
        print(f'generating {fname}...')
        data = _generate_random_shoe_reward_arrays(
            num_shoes=shoes_per_file, rng=rng, num_decks=num_decks,
            payout=payout)
        np.save(str(out_path), data)


def _generate_random_shoe_reward_array(
        row_num: int,
        num_decks: int = 3,
        payout: typing.Optional[payout_table.PayoutTable] = None) -> np.ndarray:
    cards = shoe.shuffled_cards(
        num_decks=num_decks, rng=np.random.default_rng())
    return _compute_shoe_rewards(cards, payout).as_array()


def cache_shoe_reward_arrays_parallel(
        out_dir: str,
        shoes_per_file: int,
        prefix: str = 'rewards',
        num_decks: int = 3,
        max_num_files: int = 1000,
        start_num: int = 0,
        payout: typing.Optional[payout_table.PayoutTable] = None,
        num_processes: typing.Optional[int] = None):
    payout = payout or payout_table.PayoutTable.default()
    _generate = functools.partial(
        _generate_random_shoe_reward_array, num_decks=num_decks, payout=payout)
    with multiprocessing.Pool(num_processes) as pool:
        for fnum in range(start_num, max_num_files):
            fname = f'{prefix}-nd{num_decks}-{fnum:04d}.npy'
            out_path = pathlib.Path(out_dir) / fname
            print(f'generating {fname}...')
            data_q = collections.deque()
            for row in pool.imap_unordered(_generate, range(shoes_per_file)):
                data_q.append(row)
            data = np.array(data_q)
            np.save(str(out_path), data)


# class TargetAdvantageSolver(object):
#     """Class for computing action values at each position.
#
#     Attributes:
#        hand_limit: number of hands to play, or zero for unlimited. Player
#             is assumed to play until either hand_limit hands are played or
#             shoe is depleted.
#     """
#     def __init__(self, cards: typing.Iterable[card.Card],
#                  estimator: advantage_estimator.NNAdvantageEstimator,
#                  hand_limit: int = 0,
#                  payout: typing.Optional[payout_table.PayoutTable] = None):
#         if isinstance(cards, str):
#             self.cards = card.card_list(cards)
#         else:
#             self.cards = list(cards)
#         self.estimator = estimator
#         self.hand_limit = hand_limit
#         self.payout = payout or payout_table.PayoutTable.default()
#
#     def _expected_payouts_by_num_held(self) -> typing.Tuple[
#             np.ndarray, typing.List[typing.List[hand.Hand]]]:
#         """Returns expected payout at each point in shoe for number held.
#
#         Returns an array where each element is the expected payout (i.e. if
#         you know shoe contents but not order) for a single hand starting at a
#         given point in the shoe and holding a given number of cards (discarding
#         and replacing the rest).
#
#         Returns:
#             expected: ndarray of shape (shoe_size - 10, 6), where expected[i, h]
#                 is the expected payout (given remaining shoe contents but not
#                 order) of the 5-card hand from shoe.cards[i:i+5] after
#                 discarding and replacing (5-h) cards.
#             held: 2d list of Hands, where held[i][h] is the hand that should
#                 be held when discarding (5-h) cards.
#         """
#         last_index = len(self.cards) - 10
#         expected = collections.deque()
#         held = collections.deque()
#         for i in range(last_index + 1):
#             delt = hand.Hand(self.cards[i:i+5])
#             hs = hand_stats.HandStats(
#                 cards=self.cards[i+5:], payout=self.payout)
#             exp_payout = hs.best_expected_payout_per_num_held(delt)
#             expected.append([b.value for b in exp_payout])
#             held.append([b.held for b in exp_payout])
#         return np.array(expected), list(held)
#
#     def _actual_payouts_by_num_held(
#             self, held: typing.List[typing.List[hand.Hand]]) -> np.ndarray:
#         """Returns actual payout at each point in shoe for number held.
#
#         Returns an array where each element is the actual payout for a single
#         hand starting at a given point in the shoe and holding a given number
#         of cards (discarding and replacing the rest).
#
#         Args:
#             held: 2d list of Hands, where held[i][h] is the hand that should
#                 be held when discarding (5-h) cards (e.g. as returned by
#                 _expected_payouts_by_num_held).
#
#         Returns:
#             actual: ndarray of shape (shoe_size - 10, 6), where actual[i, h]
#                 is the actual payout of the 5-card hand from shoe.cards[i:i+5]
#                 after discarding and replacing the (5-h) cards specified in
#                 held.
#         """
#         last_index = len(self.cards) - 10
#         payouts = np.zeros((last_index + 1, 6))
#         for p in range(last_index + 1):
#             replacements = self.cards[p+5:p+10]
#             for h in range(6):
#                 filled = held[p][h].add(replacements[:5-h])
#                 payouts[p, h] = filled.payout(self.payout)
#         return payouts
#
#     def get_target_advantage(self) -> typing.Tuple[
#             np.ndarray, np.ndarray, np.ndarray]:
#         """Returns states and corresponding target advantages for shoe.
#
#         Returns:
#             states: 53-float state vector
#             advantage_target: mean-normalized target advantages for hand-limit,
#                 pos. and num_held. Shape: (hand_limit, playable_shoe_size, 6)
#             advantage_target_mean: mean of advantage_target before normalization
#         """
#         # Using the current advantage estimator, compute how many cards will be
#         # held for every possible hand starting position and hand-limit in
#         # the shoe.
#
#         # Number of cards in the shoe.
#         shoe_size = len(self.cards)
#         # The window of playable hand positions in the shoe.
#         playable_shoe_size = shoe_size - 10
#         # Expected reward and num cards held: R_exp(p, a) = E[R(p, a)], H(p, a)
#         # Shape: (playable_shoe_size, 6)
#         r_expected, held = self._expected_payouts_by_num_held()
#         # Actual reward for each position and num cards held: R(p, a)
#         # Shape: (playable_shoe_size, 6)
#         r_actual = self._actual_payouts_by_num_held(held)
#         # State vectors for computing advantage: S(λ, p)
#         # Shape: (hand_limit, playable_shoe_size, 53)
#         states = advantage_estimator.norm_state_vector(
#             _shoe_to_unnormed_states(self.cards, self.hand_limit))
#         # Estimated advantage for action at given position: B_e(λ, p, a | θ)
#         # Shape: (hand_limit, playable_shoe_size, 6)
#         advantage_est = self.estimator.get_advantage_for_states(states)
#         # Estimated Q value: Q_e(λ, p, a | θ) = R_exp(p, a) + B_e(λ, p, a | θ)
#         # Shape: (hand_limit, playable_shoe_size, 6)
#         q_est = (r_expected[np.newaxis, :, :] +
#                  advantage_est.detach().cpu().numpy())
#         # Number of cards held in each state (i.e. the action taken) using the
#         # current advantage estimator.
#         # H(λ, p | θ) = argmax_a[Q_e(λ, p, a | θ)]
#         # Shape: (hand_limit, playable_shoe_size)
#         num_held = np.argmax(q_est, axis=-1)
#
#         # Now compute the value and Q-value for every hand-limit,
#         # position and num_held (action) given current params θ, iterating
#         # backwards through the shoe with increasing hand limits.
#         last_index = playable_shoe_size
#         # v_actual = np.zeros((self.hand_limit+1, shoe_size + 1)).astype(float)
#         # advantage_target = np.zeros(
#         #     (self.hand_limit+1, playable_shoe_size+1, 6))
#
#         # V(λ, p | θ): Cumulative reward starting at (λ, p) and following
#         #     policy given by θ. Padded with an initial column of zeros,
#         #     so v_actual[1, p] = V(0, p).
#         # Shape: (hand_limit + 1, shoe_size + 1)
#         v_actual = np.zeros((self.hand_limit+2, shoe_size + 1)).astype(float)
#         # A_t(λ, p | θ): Expected advantage.
#         # Shape: (hand_limit, playable_shoe_size, 6)
#         advantage_target = np.zeros(
#             (self.hand_limit+1, playable_shoe_size+1, 6))
#         for hand_limit in range(self.hand_limit + 1):
#             for p in range(last_index, -1, -1):
#                 # Q(λ, p, a | θ) = R(p, a) + V(λ-1, p + 10 - a | θ)
#                 #     actual cumulative reward when choosing a at (λ, p) and
#                 #     following the policy given by θ afterwards. Note that
#                 #     v_actual here is V for the previous hand limit.
#                 # Shape: (6,)
#                 q_actual = r_actual[p, :] + v_actual[hand_limit, p+10:p+4:-1]
#                 # V(λ, p | θ) = Q(λ, p, H(λ, p | θ))
#                 v_actual[hand_limit+1, p] = q_actual[num_held[hand_limit, p]]
#                 # A_t(λ, p | θ) ← Q(λ, p, a | θ) - R_exp(p, a)
#                 # This works because:
#                 #     E[A(λ, p | θ)] = E[Q(λ, p, a | θ) - R(p, a)]
#                 #                    = E[Q(λ, p, a | θ)] - R_exp(p, a)
#                 # We know R_exp exactly, but will substitute Q for E[Q] and
#                 # rely on sampling to drive our estimate closer to the true
#                 # estimate.
#                 advantage_target[hand_limit, p, :] = q_actual - r_expected[p, :]
#         # Zero-mean the target advantage.
#         advantage_target_mean = advantage_target.mean(axis=-1, keepdims=True)
#         advantage_target -= advantage_target_mean
#         return states, advantage_target, advantage_target_mean
#
#
# class TargetAdvantageBuffer:
#     """Buffer for holding state -> target advantage pairs.
#
#     Attributes:
#         mini_batch_size: size of each sample.
#     """
#     def __init__(self, size: int, mini_batch_size: int,
#                  rng: np.random.Generator,
#                  state_dim: int = 53, target_dim: int = 6):
#         self.mini_batch_size = mini_batch_size
#         self._state_buf = np.full(
#             (size, state_dim), np.nan, dtype=advantage_estimator.STATE_DTYPE)
#         self._target_buf = np.full(
#             (size, target_dim), np.nan, dtype=advantage_estimator.STATE_DTYPE)
#         self._next = 0
#         self.rng = rng
#
#     @property
#     def capacity(self):
#         return self._state_buf.shape[0]
#
#     @property
#     def size(self):
#         return self._next
#
#     @property
#     def num_empty_slots(self):
#         return max(0, self.capacity - self.size)
#
#     @property
#     def rng(self):
#         return self.rng
#
#     @rng.setter
#     def rng(self, rng: np.random.Generator):
#         self.rng = rng
#
#     def clear(self):
#         self._next = 0
#
#     def store(self, states: np.ndarray, targets: np.ndarray,
#               truncate: bool = False) -> bool:
#         states = states.reshape(-1, 53)
#         targets = targets.reshape(-1, 6)
#         assert states.shape[0] == targets.shape[0]
#         num_entries = states.shape[0]
#         if self.num_empty_slots < num_entries:
#             msg = (f'Buffer overflow, truncating storage of {num_entries} ' +
#                    f'entries to {self.num_empty_slots}')
#             if truncate:
#                 warnings.warn(msg)
#                 states = states[:self.num_empty_slots]
#                 targets = targets[:self.num_empty_slots]
#                 num_entries = states.shape[0]
#             else:
#                 raise ValueError(msg)
#         self._state_buf[self._next:self._next + num_entries] = states
#         self._target_buf[self._next:self._next + num_entries] = targets
#         self._next += num_entries
#         return self.num_empty_slots > 0
#
#     def sample(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
#         num_batches = self.size // self.mini_batch_size
#         batch_indexes = self.rng.choice(
#             self.size, (num_batches, self.mini_batch_size), replace=False)
#         for indexes in batch_indexes:
#             states = self._state_buf[indexes]
#             targets = self._target_buf[indexes]
#             yield states, targets
#
#     def __len__(self) -> int:
#         return self.size
#
#
# class AdvantageEstimatorTrainer(object):
#     """Trainer for AdvantageEstimator
#
#     Attribute:
#         bnet (AdvantageEstimator): model to train and select actions
#         optimizer (torch.optim): optimizer for training dqn
#         transition (list): transition information including
#                            state, action, reward, next_state, done
#     """
#     def __init__(self,
#                  # Basic configuration
#                  exp_name: str = 'PokerAdvantage',
#                  payout: typing.Optional[payout_table.PayoutTable] = None,
#                  num_decks: int = 4,
#                  log_dir: typing.Optional[str] = None,
#                  checkpoint_dir: typing.Optional[str] = None,
#                  run_name: typing.Optional[str] = None,
#                  checkpoint_interval: int = 10,
#                  # Hyperparams
#                  shoes_per_batch: int = 500,
#                  mini_batch_size: int = 52 * 4,
#                  device: str = 'auto',
#                  learning_rate: float = 2.5e-4,
#                  seed: typing.Optional[int] = None,
#                  ):
#         """Initialization."""
#         self.exp_name = exp_name
#         self.payout = payout
#         self.num_decks = num_decks
#         self.log_dir = log_dir
#         self.checkpoint_dir = checkpoint_dir
#         self.run_name = run_name
#         self.checkpoint_interval = checkpoint_interval
#
#         self.shoes_per_batch = shoes_per_batch
#         self.mini_batch_size = mini_batch_size
#         self.device = device
#         self.learning_rate = learning_rate
#         self.epoch = 0
#         self.update_num = 0
#         if seed is not None:
#             self.seed = seed
#         else:
#             self.seed = np.random.default_rng().integers(
#                 low=np.iinfo(np.int32).min, high=np.iinfo(np.int32).max)
#
#         self.rng = np.random.default_rng(seed=self.seed)
#         self._device = torch_util.torch_device(self.device)
#         self._estimator = advantage_estimator.NNAdvantageEstimator(self._device)
#         self._optimizer = optim.Adam(self._estimator.bnet.parameters(),
#                                      lr=self.learning_rate, eps=1e-5)
#         self._max_hand_limit = self.num_decks * 52 // 5 - 1
#         self._batch_size = (self.shoes_per_batch * (self.num_decks * 52 - 9) *
#                             (self._max_hand_limit + 1))
#         if self._batch_size % self.mini_batch_size:
#             ValueError(f'batch size ({self._batch_size}) must be multiple of ' +
#                        f'mini-batch size ({mini_batch_size})')
#         self._buffer = TargetAdvantageBuffer(
#             self._batch_size, self.mini_batch_size, rng=self.rng)
#
#         if not self.run_name:
#             self.run_name = f'{self.exp_name}__{self.seed}__{int(time.time())}'
#         self.writer = None
#         if self.log_dir:
#             log_path = pathlib.Path(self.log_dir) / self.run_name
#             if log_path.exists():
#                 self.writer = tensorboard.SummaryWriter(str(log_path))
#             else:
#                 self.writer = tensorboard.SummaryWriter(str(log_path))
#                 param_strs = [f'|{k}|{v}|' for k, v in self.params.items()]
#                 self.writer.add_text(
#                     'hyperparameters',
#                     '|param|value|\n|-|-|\n' + '\n'.join(param_strs))
#
#     @property
#     def params(self):
#         return dict(exp_name=self.exp_name,
#                     shoes_per_batch=self.shoes_per_batch,
#                     learning_rate=self.learning_rate,
#                     mini_batch_size=self.mini_batch_size,
#                     device=self._device,
#                     payout=self.payout,
#                     seed=self.seed,
#                     num_decks=self.num_decks)
#
#     @property
#     def rng(self):
#         return self.rng
#
#     @rng.setter
#     def rng(self, rng: np.random.Generator):
#         self.rng = rng
#         self._buffer.rng = rng
#
#     @classmethod
#     def _load_from_dict(cls,
#                         epoch: int,
#                         update_num: int,
#                         rng: typing.Optional[np.random.Generator],
#                         estimator_state_dict: typing.Dict,
#                         optimizer_state_dict: typing.Dict,
#                         **kwargs):
#         bet = AdvantageEstimatorTrainer(**kwargs)
#         bet.epoch = epoch
#         bet.update_num = update_num
#         bet.rng = rng
#         bet._estimator.load_state_dict(estimator_state_dict)
#         bet._optimizer.load_state_dict(optimizer_state_dict)
#
#     @classmethod
#     def load(cls, path: str):
#         params = torch.load(path)
#         return cls._load_from_dict(**params)
#
#     def save(self, path: str):
#         params = self.params
#         params['epoch'] = self.epoch
#         params['update_num'] = self.update_num
#         params['rng'] = self.rng
#         params['estimator_state_dict'] = (
#             self._estimator.bnet.state_dict())
#         params['optimizer_state_dict'] = self._optimizer.state_dict()
#         torch.save(params, path)
#
#     def _refill_buffer(self):
#         cards = shoe.shuffled_cards(num_decks=self.num_decks, rng=self.rng)
#         self._buffer.clear()
#         with torch.no_grad():
#             for shoe_num in range(self.shoes_per_batch):
#                 self.rng.shuffle(cards)
#                 solver = TargetAdvantageSolver(cards=cards,
#                                                estimator=self._estimator,
#                                                hand_limit=self._max_hand_limit)
#                 states, advantage_target, _ = solver.get_target_advantage()
#                 self._buffer.store(states, advantage_target)
#
#     def _refill_buffer_from_file(self, path: pathlib.Path):
#         self._buffer.clear()
#         data = np.load(str(path))
#         num_shoes = data.shape[0]
#         for shoe_num in range(num_shoes):
#             rewards = ShoeRewards.from_array(data[shoe_num])
#             with torch.no_grad():
#                 states, advantage_target, _ = get_target_advantage(
#                     rewards, hand_limit=self._max_hand_limit,
#                     estimator=self._estimator)
#                 self._buffer.store(states, advantage_target)
#
#     def update_model(self, states, targets) -> float:
#         """Update the model by gradient descent."""
#         estimated_advantage = self._estimator.get_advantage_for_states(states)
#         targets_t = torch.FloatTensor(targets).to(self._device)
#         loss = torch.abs(estimated_advantage - targets_t).sum()
#         self._optimizer.zero_grad()
#         loss.backward()
#         self._optimizer.step()
#         return loss.item()
#
#     # def _log_store(self, seconds_per_shoe_solve: float):
#     #     if self.writer:
#     #         self.writer.add_scalar(
#     #             'charts/seconds_per_shoe_solve', seconds_per_shoe_solve,
#     #             self.update_num)
#
#     def _log_update(self, loss: float, seconds_per_update: float):
#         if self.writer:
#             self.writer.add_scalar(
#                 'charts/loss_per_update', loss, self.update_num)
#             self.writer.add_scalar(
#                 'charts/seconds_per_update', seconds_per_update,
#                 self.update_num)
#
#     def _log_epoch(self, loss: float, seconds_per_store: float,
#                    seconds_per_train: float):
#         if self.writer:
#             self.writer.add_scalar(
#                 'charts/mean_loss_epoch', loss, self.epoch)
#             self.writer.add_scalar(
#                     'charts/seconds_per_epoch_store', seconds_per_store,
#                     self.epoch)
#             self.writer.add_scalar(
#                 'charts/seconds_per_epoch_train', seconds_per_train, self.epoch)
#
#     def train(self, num_epochs: int):
#         for epoch in range(num_epochs):
#             print(f'storing for epoch {self.epoch}')
#             epoch_start = time.time()
#             losses = collections.deque()
#             self._refill_buffer()
#             seconds_per_store = time.time() - epoch_start
#             print(f'storing for epoch {self.epoch} took ' +
#                   f'{seconds_per_store} seconds')
#             for states, targets in self._buffer.sample():
#                 update_start = time.time()
#                 loss = self.update_model(states, targets)
#                 seconds_per_update = time.time() - update_start
#                 losses.append(loss)
#                 self._log_update(
#                     loss=loss,
#                     seconds_per_update=seconds_per_update)
#                 self.update_num += 1
#             seconds_per_train = time.time() - epoch_start - seconds_per_store
#             self._log_epoch(np.array(losses).mean(),
#                             seconds_per_store=seconds_per_store,
#                             seconds_per_train=seconds_per_train)
#             self.epoch += 1
#
#     def train_from_files(self, num_epochs: int,
#                          data_dir: typing.Optional[str] = None):
#         files = []
#         for epoch in range(num_epochs):
#             if self.checkpoint_dir and epoch % self.checkpoint_interval == 0:
#                 out_fname = pathlib.Path(f'{self.run_name}-{epoch:04d}.zip')
#                 out_path = pathlib.Path(self.checkpoint_dir) / out_fname
#                 self.save(str(out_path))
#             epoch_start = time.time()
#             losses = collections.deque()
#             if not files:
#                 data_path = pathlib.Path(data_dir)
#                 files = [f for f in data_path.glob('*.npy')]
#                 self.rng.shuffle(files)
#             if not files:
#                 raise FileNotFoundError(f'No data files found in {data_dir}')
#             data_file = files.pop()
#             print(f'Loading {data_file} for epoch {self.epoch}')
#             self._refill_buffer_from_file(data_file)
#             seconds_per_store = time.time() - epoch_start
#             print(f'storing for epoch {self.epoch} took ' +
#                   f'{seconds_per_store} seconds')
#             for states, targets in self._buffer.sample():
#                 update_start = time.time()
#                 loss = self.update_model(states, targets)
#                 seconds_per_update = time.time() - update_start
#                 losses.append(loss)
#                 self._log_update(
#                     loss=loss,
#                     seconds_per_update=seconds_per_update)
#                 self.update_num += 1
#             seconds_per_train = time.time() - epoch_start - seconds_per_store
#             self._log_epoch(np.array(losses).mean(),
#                             seconds_per_store=seconds_per_store,
#                             seconds_per_train=seconds_per_train)
#             self.epoch += 1


if __name__ == "__main__":
    rewards_dir = '/Users/bug/Projects/GitHub/one-shoe-poker/data/rewards-1deck-65jb-ante/'
    cp_dir = '/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints-1deck-65jb-ante'
    # Standard payout for 1-deck 6/5 jacks-or-better, including ante.
    _payout = payout_table.PayoutTable(royal_flush=800, straight_flush=50,
                                      five_kind=0, four_kind=25, full_house=6,
                                      flush=5, straight=4, three_kind=3,
                                      two_pair=2, jack_high_pair=1,
                                      low_pair=0, ante=1)
    # cache_shoe_reward_arrays_parallel(rewards_dir, prefix='r1',
    #                          shoes_per_file=500, num_decks=1,
    #                          start_num=0, num_processes=7, payout=_payout)

    trainer = TableAdvantageEstimatorTrainer(
        data_dir=rewards_dir,
        num_decks=1,
        checkpoint_dir=cp_dir,
        checkpoint_interval=1,
        max_shoes_per_epoch=None)
    trainer.load_advantage(cp_dir + '/advantage-mean-1deck-latest.npy')
    tadvantage_mean, tadvantage_var, val_mean, val_var = (
        trainer.train(lr=0.1, num_epochs=1000))
    np.set_printoptions(suppress=True, precision=3)
    print('mean:')
    print(tadvantage_mean[-1])
    print('\nvariance:')
    print(tadvantage_var)




    # cp_dir = '/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints/EstTrainer/'
    # trainer = AdvantageEstimatorTrainer(
    #     exp_name='test8', num_decks=2, seed=888888,
    #     log_dir='/Users/bug/Projects/GitHub/one-shoe-poker/runs/',
    #     checkpoint_dir=cp_dir, learning_rate=.01)
    # cp_path = cp_dir + 'test8__888888__1682651080-0010.zip'
    # trainer = AdvantageEstimatorTrainer.load(
    #     cp_path, run_name=None, learning_rate=0.01)
    # trainer.train_from_files(
    #     100000,
    #     data_dir='/Users/bug/Projects/GitHub/one-shoe-poker/data/rewards/')



    # trainer = TableAdvantageEstimatorTrainer(
    #     data_dir='/Users/bug/Projects/GitHub/one-shoe-poker/data/rewards-1deck-ante/',
    #     num_decks=1,
    #     checkpoint_dir='/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints-1deck-ante/',
    #     checkpoint_interval=1,
    #     max_shoes_per_epoch=None)
    # trainer.load_advantage(
    #     '/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints-3decks/advantage-mean-3deck-latest.npy')
    # tadvantage_mean, tadvantage_var, val_mean, val_var = (
    #     trainer.train(lr=0.1, num_epochs=100))

    # print(tadvantage_mean)
    # print()
    # print(tadvantage_var)
