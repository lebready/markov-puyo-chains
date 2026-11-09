import statistics

from puyo.bots import GreedyBot, RandomBot
from puyo.engine import Board
from puyo.evaluate import bootstrap_mean_ci, play_game, run


def score_of(board: Board, pair, placement) -> int:
    after = board.copy()
    after.place(pair, placement)
    return after.resolve().score


# --- bots ----------------------------------------------------------------------


def test_random_bot_is_legal_and_reproducible():
    board = Board.from_rows(["RRR..."])
    pairs = [("R", "G"), ("B", "B"), ("Y", "R")]
    choices_a = [RandomBot(seed=1).choose(board, pairs) for _ in range(5)]
    choices_b = [RandomBot(seed=1).choose(board, pairs) for _ in range(5)]
    assert choices_a == choices_b
    assert all(c in board.legal_placements(pairs[0]) for c in choices_a)


def test_greedy_bot_takes_the_highest_scoring_move():
    # Placement(2, 0) sets off a 2-chain worth 360. Other moves pop at most
    # the reds (40).
    board = Board.from_rows(["GG....", "RRRG.."])
    pair = ("R", "G")
    choice = GreedyBot(seed=0).choose(board, [pair, pair, pair])
    assert score_of(board, pair, choice) == 360


def test_greedy_bot_avoids_topping_out():
    # Column 2 is 10 high (alternating colours, so nothing pops): a vertical
    # drop there would top out.
    board = Board([[], [], ["GB"[row % 2] for row in range(10)], [], [], []])
    pair = ("Y", "R")
    for seed in range(20):
        choice = GreedyBot(seed=seed).choose(board, [pair, pair, pair])
        after = board.copy()
        after.place(pair, choice)
        after.resolve()
        assert not after.is_dead()


# --- evaluation ----------------------------------------------------------------


def test_play_game_is_reproducible():
    a = play_game(RandomBot("bot"), seed=3, max_turns=50)
    b = play_game(RandomBot("bot"), seed=3, max_turns=50)
    assert (a.score, a.turns, a.max_chain) == (b.score, b.turns, b.max_chain)
    assert a.turns <= 50
    assert a.topped_out == (a.turns < 50)


def test_bootstrap_ci_contains_the_mean():
    values = [0, 40, 40, 360, 1000, 80, 40, 0, 120, 200]
    low, high = bootstrap_mean_ci(values)
    assert low <= statistics.fmean(values) <= high


def test_run_gives_one_summary_per_bot():
    summaries = run(["random", "greedy"], games=5, max_turns=30)
    assert [s.bot for s in summaries] == ["random", "greedy"]
    assert all(s.games == 5 for s in summaries)
