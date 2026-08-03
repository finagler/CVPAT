from agent import player_agent
from game import payout_table
from game import poker_game
from game import hand
from game import shoe


def test_play_game_hand_stats():
    game = poker_game.PokerGame(num_decks=1, hand_limit=0, payout=None)
    agent = player_agent.CardCountingAgent(verbose=True, name='ps')
    trajectory = agent.play(game)
    assert len(trajectory) > 3
    assert len(trajectory[0].info) == 1
    assert trajectory[0].info[0]['name'] == 'ps'


def test_play_game_psychic_hand_limit():
    cards = ('JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S ' +
             '6C 9C 2D 8H KC KD KH 3C 4S 6S')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0, 0)
    game = poker_game.PokerGame(num_decks=1, hand_limit=2, payout=payout,
                                draw_shoe=shoe.Shoe(cards))
    agent = player_agent.PsychicAgent(verbose=True, name='psy')
    trajectory = agent.play(game)
    expected = [
        poker_game.GameMove(
            hand.Hand('JC JD 8C JH 8D'),
            hand.Hand('JC JD JH'),
            hand.Hand('JC JD JH 2C 3D'),
            2,
            (dict(name='psy', held=hand.Hand('JC JD JH'),
                  cum_value=1002., value=2.),)),
        poker_game.GameMove(
            hand.Hand('AS KS QS JS 4C'),
            hand.Hand('AS KS QS JS'),
            hand.Hand('AS KS QS JS 10S'),
            1000,
            (dict(name='psy', held=hand.Hand('AS KS QS JS'),
                  cum_value=1000., value=1000),))]
    assert trajectory == expected


def test_play_game_psychic_unlimited():
    cards = ('JC JD 8C JH 8D 2C 3D AS KS QS JS 4C 10S ' +
             '6C 9C 2D 8H KC KD KH 3C 4S 6S')
    payout = payout_table.PayoutTable(1000, 100, 20, 8, 7, 5, 4, 2, 1, 1, 0, 0)
    game = poker_game.PokerGame(num_decks=1, hand_limit=0, payout=payout,
                                draw_shoe=shoe.Shoe(cards))
    agent = player_agent.PsychicAgent(verbose=True, name='psy')
    trajectory = agent.play(game)
    expected = [
        poker_game.GameMove(
            hand.Hand('JC JD 8C JH 8D'),
            hand.Hand('JC JD JH'),
            hand.Hand('JC JD JH 2C 3D'),
            2,
            (dict(name='psy', held=hand.Hand('JC JD JH'),
                  cum_value=1004., value=2.),)),
        poker_game.GameMove(
            hand.Hand('AS KS QS JS 4C'),
            hand.Hand('AS KS QS JS'),
            hand.Hand('AS KS QS JS 10S'),
            1000,
            (dict(name='psy', held=hand.Hand('AS KS QS JS'),
                  cum_value=1002., value=1000.),)),
        poker_game.GameMove(
            hand.Hand('6C 9C 2D 8H KC'),
            hand.Hand('KC'),
            hand.Hand('KC KD KH 3C 4S'),
            2,
            (dict(name='psy', held=hand.Hand('KC'),
                  cum_value=2., value=2.),))]

    assert trajectory == expected


def test_play_game_multiple_agents():
    game = poker_game.PokerGame(num_decks=1, hand_limit=0, payout=None)
    hs_agent = player_agent.CardCountingAgent(verbose=True, name='ps')
    psy_agent = player_agent.PsychicAgent(verbose=True, name='psy')
    trajectory = player_agent.play_game(game, [hs_agent, psy_agent])
    assert len(trajectory) > 3
    assert len(trajectory[0].info) == 2
    assert trajectory[0].info[0]['name'] == 'ps'
    assert trajectory[0].info[1]['name'] == 'psy'


def test_play_game_hand_limit():
    game = poker_game.PokerGame(num_decks=1, hand_limit=2, payout=None)
    psy_agent = player_agent.PsychicAgent(verbose=True, name='psy')
    hs_agent = player_agent.CardCountingAgent(
        verbose=True, name='ps')
    trajectory = player_agent.play_game(game, [psy_agent, hs_agent])
    assert len(trajectory) == 2


def test_play_game_no_short_shoe():
    card_list = ('AS 2S 3S 4S 5S ' +
                 '6S 7S 8S 9S 10S ' +
                 '5C 6C 7C 8C 9C ' +
                 'AC 3D 5H 7S 9C ' +
                 'AS 7C')
    draw_shoe = shoe.Shoe(card_list=card_list)
    game = poker_game.PokerGame(num_decks=1, hand_limit=0, payout=None,
                                draw_shoe=draw_shoe, allow_short_shoe=False)
    psy_agent = player_agent.PsychicAgent(verbose=True, name='psy')
    hs_agent = player_agent.CardCountingAgent(verbose=True, name='ps')
    trajectory = player_agent.play_game(game, [psy_agent, hs_agent])
    assert len(trajectory) == 3


def test_play_game_short_shoe():
    card_list = ('AS 2S 3S 4S 5S ' +
                 '6S 7S 8S 9S 10S ' +
                 '5C 6C 7C 8C 9C ' +
                 'AC 3D 5H 7S 9C ' +
                 'AS 7C')
    draw_shoe = shoe.Shoe(card_list=card_list)
    game = poker_game.PokerGame(num_decks=1, hand_limit=0, payout=None,
                                draw_shoe=draw_shoe, allow_short_shoe=True)
    psy_agent = player_agent.PsychicAgent(verbose=True, name='psy')
    hs_agent = player_agent.CardCountingAgent(verbose=True, name='ps')
    trajectory = player_agent.play_game(game, [psy_agent, hs_agent])
    assert len(trajectory) == 4


def test_simple_player_play_game():
    game = poker_game.PokerGame(num_decks=4, hand_limit=0, payout=None)
    agent = player_agent.SimplePlayerAgent(verbose=True, name='simple')
    trajectory = agent.play(game)
    assert len(trajectory) > 3
    assert len(trajectory[0].info) == 1
    assert trajectory[0].info[0]['name'] == 'simple'


def test_simple_player_get_move_no_value():
    game = poker_game.PokerGame(num_decks=4, hand_limit=0, payout=None)
    agent = player_agent.SimplePlayerAgent(verbose=True, name='simple')
    game.hand = hand.Hand('AC KD 2H 5S 10C')
    held, info = agent.get_move(game)
    assert held == hand.Hand('AC KD')
    assert info['current_payout'] == 0


def test_simple_player_get_move_3kind():
    payout = payout_table.PayoutTable(
        10000, 1000, 200, 80, 70, 60, 50, 20, 15, 10, 0, 0)
    game = poker_game.PokerGame(num_decks=4, hand_limit=0, payout=payout)
    agent = player_agent.SimplePlayerAgent(verbose=True, name='simple')
    game.hand = hand.Hand('2C 2D 2H 5S KC')
    held, info = agent.get_move(game)
    assert held == hand.Hand('2C 2D 2H')
    assert info['current_payout'] == 20
