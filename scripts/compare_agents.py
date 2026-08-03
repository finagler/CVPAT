import typing

import numpy as np
from scipy import stats

from game import poker_game
from agent import player_agent
from game import payout_table


def ci(data: np.ndarray, confidence: float = 0.95) -> float:
    """Returns confidence interval of estimated mean.

    Args:
        data: values for which mean is to be estimated.
        confidence: probability that reported mean is within returned interval.

    Returns:
        interval: mean of data = mean ± interval, with given confidence.
    """
    mean = data.mean()
    try:
        interval = stats.norm.interval(
            confidence, loc=mean, scale=stats.sem(data))
    except FloatingPointError:
        return np.nan
    return interval[1] - mean


def print_mean_payouts(agents: typing.List[player_agent.PlayerAgent],
                       num_trials: int = 10, num_decks: int = 4,
                       hand_limit: int = 0):
    payout = payout_table.PayoutTable.default()
    payouts = np.full((num_trials, len(agents)), np.nan)

    for trial in range(num_trials):
        if trial % 10 == 0:
            print(f'trial: {trial}')
        game = poker_game.PokerGame(
            num_decks=num_decks, payout=payout, hand_limit=hand_limit)
        for anum, agent in enumerate(agents):
            traj = agent.play(game.copy())
            payouts[trial, anum] = np.array(
                [move.payout for move in traj]).sum()
    for anum, agent in enumerate(agents):
        row = payouts[:, anum]
        print(f'{agent.name} mean total payout: {row.mean()}  ± {ci(row):.2f}')
    pct_win = (payouts[:, 0] > payouts[:, 1]).sum() / num_trials * 100
    pct_tie = (payouts[:, 0] == payouts[:, 1]).sum() / num_trials * 100
    pct_lose = (payouts[:, 0] < payouts[:, 1]).sum() / num_trials * 100
    print(f'{agents[0].name} > {agents[1].name}: {pct_win}')
    print(f'{agents[0].name} == {agents[1].name}: {pct_tie}')
    print(f'{agents[0].name} < {agents[1].name}: {pct_lose}')


if __name__ == '__main__':
    advantage1 = np.load(
        '/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints/advantage-2deck.npy')
    advantage2 = np.load(
        '/Users/bug/Projects/GitHub/one-shoe-poker/checkpoints/advantage-mean-2deck-00094.npy')
    my_agents = [player_agent.CardCountingAgent(verbose=False, name='base'),
                 player_agent.CardCountingAgent(verbose=False, name='advantage_2',
                                                advantage=advantage2),
                 player_agent.CardCountingAgent(verbose=False, name='advantage_1',
                                                advantage=advantage1)]
    print_mean_payouts(my_agents, num_trials=1000, num_decks=2)

