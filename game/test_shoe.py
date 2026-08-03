from unittest import mock

from game import shoe


def test_draw_cards():
    draw_shoe = shoe.Shoe()
    first_five = draw_shoe.cards[:5]
    rest_of_shoe = draw_shoe.cards[5:]
    draw = draw_shoe.draw_cards()
    assert draw == first_five
    assert draw_shoe.cards == rest_of_shoe


def test_shoe_with_num_cards_remaining():
    draw_shoe = shoe.Shoe.shoe_with_num_cards_remaining(num_cards=10)
    assert draw_shoe.size() == 10


@mock.patch('game.shoe.input', create=True)
def test_interactive_shoe(mocked_input):
    mocked_input.side_effect = ['AC 2C 3C 4C 5C']
    draw_shoe = shoe.InteractiveShoe(num_decks=2)
    draw_shoe.draw_cards(5)
    assert draw_shoe.size() == 99
