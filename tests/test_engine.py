import random

import pytest

from puyo.engine import (
    HEIGHT,
    Board,
    Game,
    Placement,
    step_score,
)


def filler(column: int, height: int) -> list[str]:
    """A column of alternating colours that never forms a group with its
    neighbours (a checkerboard pattern)."""
    return ["GB"[(row + column) % 2] for row in range(height)]


# --- board text format ---------------------------------------------------------


def test_from_rows_round_trip():
    board = Board.from_rows(["G.....", "RRB..Y"])
    assert board.get(0, 0) == "R"
    assert board.get(0, 1) == "G"
    assert board.height(5) == 1
    assert Board.from_rows(board.to_rows()) == board
    assert len(board.to_rows()) == HEIGHT


def test_from_rows_rejects_floating_puyo():
    with pytest.raises(ValueError, match="floating"):
        Board.from_rows(["R.....", "......"])


# --- placements ----------------------------------------------------------------


def test_empty_board_has_22_placements_or_11_for_same_colour():
    board = Board()
    assert len(board.legal_placements(("R", "G"))) == 22
    assert len(board.legal_placements(("R", "R"))) == 11


def test_horizontal_pair_splits():
    board = Board.from_rows(["R.....", "G....."])
    board.place(("B", "Y"), Placement(0, 1))  # axis on column 0, child to the right
    assert board.get(0, 2) == "B"
    assert board.get(1, 0) == "Y"


def test_rotation_2_puts_child_below():
    board = Board()
    board.place(("B", "Y"), Placement(4, 2))
    assert board.columns[4] == ["Y", "B"]


def test_tall_column_blocks_movement_past_it():
    board = Board([[], filler(1, 12), [], [], [], []])
    placements = board.legal_placements(("R", "G"))
    # Columns 2-5 only: 8 vertical + 6 horizontal.
    assert len(placements) == 14
    assert Placement(0, 0) not in placements
    assert Placement(2, 3) not in placements  # child would go into column 1


# --- scoring -------------------------------------------------------------------


def test_step_score_formula():
    assert step_score(1, [4], 1) == 40  # bracket 0 is clamped up to 1
    assert step_score(2, [4], 1) == 320  # 10 x 4 x 8
    assert step_score(1, [5], 1) == 100  # group bonus 2
    assert step_score(1, [4, 4], 2) == 240  # colour bonus 3
    assert step_score(30, [11] * 50, 5) == 10 * 550 * 999  # bracket clamped to 999


def test_single_pop():
    board = Board.from_rows(["RRR..."])
    board.place(("R", "G"), Placement(3, 0))
    result = board.resolve()
    assert result.chain_length == 1
    assert result.score == 40
    assert board == Board.from_rows(["...G.."])
    assert not result.all_clear


def test_two_chain_and_all_clear():
    # The reds pop first. The green on top of column 2 then falls next to the
    # other greens and pops as the second step.
    board = Board.from_rows(["GG....", "RRRG.."])
    board.place(("R", "G"), Placement(2, 0))
    result = board.resolve()
    assert result.chain_length == 2
    assert [step.score for step in result.steps] == [40, 320]
    assert result.score == 360
    assert result.all_clear


def test_two_colours_popping_at_once():
    board = Board.from_rows([".....G", "RRR.GG"])
    board.place(("R", "G"), Placement(3, 1))
    result = board.resolve()
    assert result.chain_length == 1
    assert result.steps[0].colors == frozenset("RG")
    assert result.score == 240


def test_group_of_five():
    board = Board.from_rows(["RR.RR."])
    board.place(("R", "G"), Placement(2, 0))
    result = board.resolve()
    assert result.steps[0].group_sizes == (5,)
    assert result.score == 100


# --- hidden 13th row and death -------------------------------------------------


def test_hidden_row_does_not_pop():
    # Three reds in the top visible row, a fourth red in the hidden row above.
    columns = [[], [], [], filler(3, 11) + ["R"], filler(4, 11) + ["R"], filler(5, 11) + ["R", "R"]]
    board = Board(columns)
    assert board.resolve().chain_length == 0


def test_hidden_puyo_falls_after_a_pop():
    # Same as above plus a fourth visible red, so the visible reds pop and
    # the hidden red drops into row 11.
    columns = [[], [], filler(2, 11) + ["R"], filler(3, 11) + ["R"], filler(4, 11) + ["R"], filler(5, 11) + ["R", "R"]]
    board = Board(columns)
    result = board.resolve()
    assert result.chain_length == 1
    assert board.get(5, 11) == "R"
    assert board.height(5) == 12


def test_death_only_in_spawn_column():
    assert Board([[], [], filler(2, 12), [], [], []]).is_dead()
    assert not Board([[], filler(1, 12), [], [], [], []]).is_dead()


# --- game ----------------------------------------------------------------------


def test_same_seed_same_pairs_and_three_visible():
    a, b = Game(seed=7), Game(seed=7)
    assert a.visible_pairs() == b.visible_pairs()
    assert len(a.visible_pairs()) == 3


def test_unreachable_placement_is_rejected():
    game = Game(seed=0)
    game.board = Board([[], filler(1, 12), [], [], [], []])
    with pytest.raises(ValueError):
        game.play(Placement(0, 0))


def test_random_game_runs_to_the_end():
    rng = random.Random(0)
    game = Game(seed=0, max_turns=300)
    while not game.is_over:
        game.play(rng.choice(game.legal_placements()))
        assert game.board.find_groups() == []
        assert all(len(col) <= HEIGHT for col in game.board.columns)
    assert game.is_over
    assert game.legal_placements() == []
