"""Interface for aiding or playing an external game."""
import numpy as np

from agent import player_agent
from game import shoe
from game import poker_game
from game import payout_table

NUM_DECKS = 3


def play_game(num_decks: int = 4):
    np.set_printoptions(precision=1, suppress=True)
    advantage = None
    try:
        advantage = player_agent.advantage_table(num_decks=num_decks)
    except FileNotFoundError:
        print(f'Cannot find advantage table for {num_decks} decks, playing ',
              'without advantage')
    agent = player_agent.CardCountingAgent(advantage=advantage, verbose=True)
    draw_shoe = shoe.InteractiveShoe(num_decks=num_decks)
    payout = payout_table.PayoutTable.default()
    game = poker_game.PokerGame(num_decks=num_decks, draw_shoe=draw_shoe,
                                payout=payout, allow_short_shoe=True)
    try:
        agent.play(game, verbose=True)
    except KeyboardInterrupt:
        print('Done.')


if __name__ == '__main__':
    play_game(num_decks=NUM_DECKS)
