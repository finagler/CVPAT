import numpy as np

from stats import shoe_stats
from agent import player_agent


def test_calc_score_distribution():
    cards_str = ('4H 9D QD 4C 3S KH 9S 8D JS 2H KS 3D 5C JS 5C 7H AH 3C 4D AD '
                 'JH 2H AS 6D 2D 7D 9H KS AC 5S 7S 8C 10C 7S AD KC 6S 3D 9H 5D '
                 'QH 8S 10D 6H AH 5S JC 10H 7D 7H 10C 5H AC 7C 4C QC 3H 3H 9D '
                 '3C 9S QC JD 2S KH 2C 9C 4S 4H 6C JC 8H 2C JH 6D 3S 10S 10D '
                 '4D 6H 5H 4S 9C QS KD KC KD 5D 2D 8S 8C 7C 6C JD QS 10H 10S '
                 '6S 8D 8H AS QH 2S QD')
    norm, dist, total = shoe_stats.calc_score_distribution(cards_str)
    assert norm[365] == 1.0
    assert total == 517977245757454909896


def test_analyze_json_str():
    json_str = """
     [{"seed":752596,"deck":
        [{"rank":7,"suit":8},{"rank":2,"suit":1},{"rank":12,"suit":16},
         {"rank":13,"suit":16},{"rank":10,"suit":1},{"rank":9,"suit":1},
         {"rank":13,"suit":1},{"rank":7,"suit":4},{"rank":9,"suit":4},
         {"rank":3,"suit":8},{"rank":3,"suit":16},{"rank":3,"suit":4},
         {"rank":12,"suit":1},{"rank":8,"suit":1},{"rank":13,"suit":1},
         {"rank":11,"suit":8},{"rank":3,"suit":1},{"rank":6,"suit":4},
         {"rank":5,"suit":16},{"rank":8,"suit":16},{"rank":11,"suit":4},
         {"rank":6,"suit":8},{"rank":11,"suit":1},{"rank":13,"suit":8},
         {"rank":6,"suit":1},{"rank":8,"suit":1},{"rank":10,"suit":4},
         {"rank":2,"suit":8},{"rank":4,"suit":1},{"rank":2,"suit":4},
         {"rank":6,"suit":8},{"rank":2,"suit":4},{"rank":8,"suit":8},
         {"rank":11,"suit":16},{"rank":3,"suit":16},{"rank":12,"suit":16},
         {"rank":14,"suit":4},{"rank":9,"suit":1},{"rank":4,"suit":8},
         {"rank":7,"suit":1},{"rank":5,"suit":16},{"rank":11,"suit":1},
         {"rank":6,"suit":4},{"rank":8,"suit":4},{"rank":12,"suit":1},
         {"rank":8,"suit":8},{"rank":2,"suit":16},{"rank":9,"suit":8},
         {"rank":12,"suit":4},{"rank":5,"suit":4},{"rank":10,"suit":8},
         {"rank":5,"suit":1},{"rank":9,"suit":4},{"rank":4,"suit":16},
         {"rank":3,"suit":8},{"rank":11,"suit":16},{"rank":13,"suit":4},
         {"rank":14,"suit":8},{"rank":11,"suit":8},{"rank":14,"suit":1},
         {"rank":12,"suit":8},{"rank":7,"suit":8},{"rank":5,"suit":8},
         {"rank":7,"suit":4},{"rank":10,"suit":16},{"rank":11,"suit":4},
         {"rank":9,"suit":8},{"rank":9,"suit":16},{"rank":7,"suit":16},
         {"rank":13,"suit":4},{"rank":9,"suit":16},{"rank":6,"suit":16},
         {"rank":14,"suit":4},{"rank":14,"suit":8},{"rank":2,"suit":1},
         {"rank":10,"suit":8},{"rank":12,"suit":4},{"rank":4,"suit":1},
         {"rank":10,"suit":4},{"rank":2,"suit":8},{"rank":13,"suit":8},
         {"rank":6,"suit":16},{"rank":8,"suit":16},{"rank":14,"suit":1},
         {"rank":12,"suit":8},{"rank":4,"suit":4},{"rank":13,"suit":16},
         {"rank":7,"suit":16},{"rank":4,"suit":4},{"rank":8,"suit":4},
         {"rank":10,"suit":1},{"rank":5,"suit":4},{"rank":7,"suit":1},
         {"rank":6,"suit":1},{"rank":2,"suit":16},{"rank":10,"suit":16},
         {"rank":4,"suit":16},{"rank":3,"suit":4},{"rank":4,"suit":8},
         {"rank":14,"suit":16},{"rank":5,"suit":8},{"rank":14,"suit":16},
         {"rank":3,"suit":1},{"rank":5,"suit":1}]}]"""
    agents = [player_agent.SimplePlayerAgent(), player_agent.HandOnlyAgent(),
              player_agent.CardCountingAgent(), player_agent.PsychicAgent()]
    score = shoe_stats.analyze_json_list_str(json_str, agents)
    assert score == [(752596, [85, 145, 145, 440])]


def test_analyze_json_str_extended():
    json_str = """
     [{"seed":752596,"deck":
        [{"rank":7,"suit":8},{"rank":2,"suit":1},{"rank":12,"suit":16},
         {"rank":13,"suit":16},{"rank":10,"suit":1},{"rank":9,"suit":1},
         {"rank":13,"suit":1},{"rank":7,"suit":4},{"rank":9,"suit":4},
         {"rank":3,"suit":8},{"rank":3,"suit":16},{"rank":3,"suit":4},
         {"rank":12,"suit":1},{"rank":8,"suit":1},{"rank":13,"suit":1},
         {"rank":11,"suit":8},{"rank":3,"suit":1},{"rank":6,"suit":4},
         {"rank":5,"suit":16},{"rank":8,"suit":16},{"rank":11,"suit":4},
         {"rank":6,"suit":8},{"rank":11,"suit":1},{"rank":13,"suit":8},
         {"rank":6,"suit":1},{"rank":8,"suit":1},{"rank":10,"suit":4},
         {"rank":2,"suit":8},{"rank":4,"suit":1},{"rank":2,"suit":4},
         {"rank":6,"suit":8},{"rank":2,"suit":4},{"rank":8,"suit":8},
         {"rank":11,"suit":16},{"rank":3,"suit":16},{"rank":12,"suit":16},
         {"rank":14,"suit":4},{"rank":9,"suit":1},{"rank":4,"suit":8},
         {"rank":7,"suit":1},{"rank":5,"suit":16},{"rank":11,"suit":1},
         {"rank":6,"suit":4},{"rank":8,"suit":4},{"rank":12,"suit":1},
         {"rank":8,"suit":8},{"rank":2,"suit":16},{"rank":9,"suit":8},
         {"rank":12,"suit":4},{"rank":5,"suit":4},{"rank":10,"suit":8},
         {"rank":5,"suit":1},{"rank":9,"suit":4},{"rank":4,"suit":16},
         {"rank":3,"suit":8},{"rank":11,"suit":16},{"rank":13,"suit":4},
         {"rank":14,"suit":8},{"rank":11,"suit":8},{"rank":14,"suit":1},
         {"rank":12,"suit":8},{"rank":7,"suit":8},{"rank":5,"suit":8},
         {"rank":7,"suit":4},{"rank":10,"suit":16},{"rank":11,"suit":4},
         {"rank":9,"suit":8},{"rank":9,"suit":16},{"rank":7,"suit":16},
         {"rank":13,"suit":4},{"rank":9,"suit":16},{"rank":6,"suit":16},
         {"rank":14,"suit":4},{"rank":14,"suit":8},{"rank":2,"suit":1},
         {"rank":10,"suit":8},{"rank":12,"suit":4},{"rank":4,"suit":1},
         {"rank":10,"suit":4},{"rank":2,"suit":8},{"rank":13,"suit":8},
         {"rank":6,"suit":16},{"rank":8,"suit":16},{"rank":14,"suit":1},
         {"rank":12,"suit":8},{"rank":4,"suit":4},{"rank":13,"suit":16},
         {"rank":7,"suit":16},{"rank":4,"suit":4},{"rank":8,"suit":4},
         {"rank":10,"suit":1},{"rank":5,"suit":4},{"rank":7,"suit":1},
         {"rank":6,"suit":1},{"rank":2,"suit":16},{"rank":10,"suit":16},
         {"rank":4,"suit":16},{"rank":3,"suit":4},{"rank":4,"suit":8},
         {"rank":14,"suit":16},{"rank":5,"suit":8},{"rank":14,"suit":16},
         {"rank":3,"suit":1},{"rank":5,"suit":1}]}]"""
    score = shoe_stats.analyze_json_list_str(
        json_str, seed=np.random.SeedSequence(entropy=12345))
    assert score == [(752596, [85, 145, 145, 130, 145, 70, 440])]


def test_analyze_json_str_list():
    # Note: on OSX PyCharm debugger complains about Error loading
    # attach_x86_64.dylib. See https://youtrack.jetbrains.com/issue/PY-48163
    json_str = """
         {"seed":752596,"deck":
            [{"rank":7,"suit":8},{"rank":2,"suit":1},{"rank":12,"suit":16},
             {"rank":13,"suit":16},{"rank":10,"suit":1},{"rank":9,"suit":1},
             {"rank":13,"suit":1},{"rank":7,"suit":4},{"rank":9,"suit":4},
             {"rank":3,"suit":8},{"rank":3,"suit":16},{"rank":3,"suit":4},
             {"rank":12,"suit":1},{"rank":8,"suit":1},{"rank":13,"suit":1},
             {"rank":11,"suit":8},{"rank":3,"suit":1},{"rank":6,"suit":4},
             {"rank":5,"suit":16},{"rank":8,"suit":16},{"rank":11,"suit":4},
             {"rank":6,"suit":8},{"rank":11,"suit":1},{"rank":13,"suit":8},
             {"rank":6,"suit":1},{"rank":8,"suit":1},{"rank":10,"suit":4},
             {"rank":2,"suit":8},{"rank":4,"suit":1},{"rank":2,"suit":4},
             {"rank":6,"suit":8},{"rank":2,"suit":4},{"rank":8,"suit":8},
             {"rank":11,"suit":16},{"rank":3,"suit":16},{"rank":12,"suit":16},
             {"rank":14,"suit":4},{"rank":9,"suit":1},{"rank":4,"suit":8},
             {"rank":7,"suit":1},{"rank":5,"suit":16},{"rank":11,"suit":1},
             {"rank":6,"suit":4},{"rank":8,"suit":4},{"rank":12,"suit":1},
             {"rank":8,"suit":8},{"rank":2,"suit":16},{"rank":9,"suit":8},
             {"rank":12,"suit":4},{"rank":5,"suit":4},{"rank":10,"suit":8},
             {"rank":5,"suit":1},{"rank":9,"suit":4},{"rank":4,"suit":16},
             {"rank":3,"suit":8},{"rank":11,"suit":16},{"rank":13,"suit":4},
             {"rank":14,"suit":8},{"rank":11,"suit":8},{"rank":14,"suit":1},
             {"rank":12,"suit":8},{"rank":7,"suit":8},{"rank":5,"suit":8},
             {"rank":7,"suit":4},{"rank":10,"suit":16},{"rank":11,"suit":4},
             {"rank":9,"suit":8},{"rank":9,"suit":16},{"rank":7,"suit":16},
             {"rank":13,"suit":4},{"rank":9,"suit":16},{"rank":6,"suit":16},
             {"rank":14,"suit":4},{"rank":14,"suit":8},{"rank":2,"suit":1},
             {"rank":10,"suit":8},{"rank":12,"suit":4},{"rank":4,"suit":1},
             {"rank":10,"suit":4},{"rank":2,"suit":8},{"rank":13,"suit":8},
             {"rank":6,"suit":16},{"rank":8,"suit":16},{"rank":14,"suit":1},
             {"rank":12,"suit":8},{"rank":4,"suit":4},{"rank":13,"suit":16},
             {"rank":7,"suit":16},{"rank":4,"suit":4},{"rank":8,"suit":4},
             {"rank":10,"suit":1},{"rank":5,"suit":4},{"rank":7,"suit":1},
             {"rank":6,"suit":1},{"rank":2,"suit":16},{"rank":10,"suit":16},
             {"rank":4,"suit":16},{"rank":3,"suit":4},{"rank":4,"suit":8},
             {"rank":14,"suit":16},{"rank":5,"suit":8},{"rank":14,"suit":16},
             {"rank":3,"suit":1},{"rank":5,"suit":1}]}"""
    json_str_list = f'[{json_str}, {json_str}, {json_str}]'
    agents = [
        player_agent.SimplePlayerAgent(),
        player_agent.HandOnlyAgent(),
        player_agent.CardCountingAgent(),
        player_agent.PsychicAgent()]
    scores = shoe_stats.analyze_json_list_str(
        json_str_list, agents=agents, num_processes=3)
    expected_score = (752596, [85, 145, 145, 440])
    assert scores == [expected_score, expected_score, expected_score]



def test_analyze_json_str_list_extended():
    # Note: on OSX PyCharm debugger complains about Error loading
    # attach_x86_64.dylib. See https://youtrack.jetbrains.com/issue/PY-48163
    json_str = """
         {"seed":752596,"deck":
            [{"rank":7,"suit":8},{"rank":2,"suit":1},{"rank":12,"suit":16},
             {"rank":13,"suit":16},{"rank":10,"suit":1},{"rank":9,"suit":1},
             {"rank":13,"suit":1},{"rank":7,"suit":4},{"rank":9,"suit":4},
             {"rank":3,"suit":8},{"rank":3,"suit":16},{"rank":3,"suit":4},
             {"rank":12,"suit":1},{"rank":8,"suit":1},{"rank":13,"suit":1},
             {"rank":11,"suit":8},{"rank":3,"suit":1},{"rank":6,"suit":4},
             {"rank":5,"suit":16},{"rank":8,"suit":16},{"rank":11,"suit":4},
             {"rank":6,"suit":8},{"rank":11,"suit":1},{"rank":13,"suit":8},
             {"rank":6,"suit":1},{"rank":8,"suit":1},{"rank":10,"suit":4},
             {"rank":2,"suit":8},{"rank":4,"suit":1},{"rank":2,"suit":4},
             {"rank":6,"suit":8},{"rank":2,"suit":4},{"rank":8,"suit":8},
             {"rank":11,"suit":16},{"rank":3,"suit":16},{"rank":12,"suit":16},
             {"rank":14,"suit":4},{"rank":9,"suit":1},{"rank":4,"suit":8},
             {"rank":7,"suit":1},{"rank":5,"suit":16},{"rank":11,"suit":1},
             {"rank":6,"suit":4},{"rank":8,"suit":4},{"rank":12,"suit":1},
             {"rank":8,"suit":8},{"rank":2,"suit":16},{"rank":9,"suit":8},
             {"rank":12,"suit":4},{"rank":5,"suit":4},{"rank":10,"suit":8},
             {"rank":5,"suit":1},{"rank":9,"suit":4},{"rank":4,"suit":16},
             {"rank":3,"suit":8},{"rank":11,"suit":16},{"rank":13,"suit":4},
             {"rank":14,"suit":8},{"rank":11,"suit":8},{"rank":14,"suit":1},
             {"rank":12,"suit":8},{"rank":7,"suit":8},{"rank":5,"suit":8},
             {"rank":7,"suit":4},{"rank":10,"suit":16},{"rank":11,"suit":4},
             {"rank":9,"suit":8},{"rank":9,"suit":16},{"rank":7,"suit":16},
             {"rank":13,"suit":4},{"rank":9,"suit":16},{"rank":6,"suit":16},
             {"rank":14,"suit":4},{"rank":14,"suit":8},{"rank":2,"suit":1},
             {"rank":10,"suit":8},{"rank":12,"suit":4},{"rank":4,"suit":1},
             {"rank":10,"suit":4},{"rank":2,"suit":8},{"rank":13,"suit":8},
             {"rank":6,"suit":16},{"rank":8,"suit":16},{"rank":14,"suit":1},
             {"rank":12,"suit":8},{"rank":4,"suit":4},{"rank":13,"suit":16},
             {"rank":7,"suit":16},{"rank":4,"suit":4},{"rank":8,"suit":4},
             {"rank":10,"suit":1},{"rank":5,"suit":4},{"rank":7,"suit":1},
             {"rank":6,"suit":1},{"rank":2,"suit":16},{"rank":10,"suit":16},
             {"rank":4,"suit":16},{"rank":3,"suit":4},{"rank":4,"suit":8},
             {"rank":14,"suit":16},{"rank":5,"suit":8},{"rank":14,"suit":16},
             {"rank":3,"suit":1},{"rank":5,"suit":1}]}"""
    json_str_list = f'[{json_str}, {json_str}, {json_str}]'
    agents = shoe_stats.default_target_agents(num_decks=3)
    seed = np.random.SeedSequence(entropy=12345)
    scores = shoe_stats.analyze_json_list_str(
        json_str_list, agents=agents, num_processes=3, ranks=(2, 3, 5, 6),
        seed=seed)
    assert scores == [(752596, [85, 95, 135, 145]),
                      (752596, [85, 95, 135, 145]),
                      (752596, [85, 95, 135, 145])]
    all_scores = shoe_stats.analyze_json_list_str(
        json_str_list, agents=agents, num_processes=3, seed=seed)
    assert all_scores == [
        (752596, [85, 145, 145, 95, 145, 70, 440]),
        (752596, [85, 145, 145, 95, 145, 70, 440]),
        (752596, [85, 145, 145, 95, 145, 70, 440])]


def test_scores_to_json_str():
    scores = [(100000, [100, 200, 300, 400]),
              (100001, [110, 220, 330, 440])]
    scores_json = shoe_stats.scores_to_json_str(scores)
    assert scores_json == (
        '[{"100000": [100, 200, 300, 400]}, {"100001": [110, 220, 330, 440]}]')
