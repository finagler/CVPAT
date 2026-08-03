import pytest

from game import poker_game
from game import card
from game import shoe


def test_error_shoe_size_ne_num_decks():
    with pytest.raises(ValueError):
        poker_game.PokerGame(num_decks=1, draw_shoe=shoe.Shoe(num_decks=2))


def test_init_does_not_draw_hand():
    game = poker_game.PokerGame(num_decks=1)
    assert game.draw_shoe.size() == 52
    assert game.hand.size() == 0


def test_not_terminate_10_cards():
    cards = card.card_list('AS 2S 3S 4S 5S 6S 7S 8S 9S 10S')
    game = poker_game.PokerGame(num_decks=1, draw_shoe=shoe.Shoe(cards))
    assert not game.draw_new_hand()

def test_terminate_9_cards():
    cards = card.card_list('AS 2S 3S 4S 5S 6S 7S 8S 9S')
    game = poker_game.PokerGame(num_decks=1, draw_shoe=shoe.Shoe(cards))
    assert game.draw_new_hand()


def test_not_terminate_5_cards_allow_short_shoe():
    cards = card.card_list('AS 2S 3S 4S 5S')
    game = poker_game.PokerGame(num_decks=1, draw_shoe=shoe.Shoe(cards),
                                allow_short_shoe=True)
    assert not game.draw_new_hand()


def test_terminate_4_cards_allow_short_shoe():
    cards = card.card_list('AS 2S 3S 4S')
    game = poker_game.PokerGame(num_decks=1, draw_shoe=shoe.Shoe(cards),
                                allow_short_shoe=True)
    assert game.draw_new_hand()


