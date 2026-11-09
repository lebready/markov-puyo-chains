import random

from puyo.engine import COLORS, VISIBLE_HEIGHT, WIDTH, Board, Game
from puyo.potential import best_potential_chain, find_potential_chains


def filler(column: int, height: int) -> list[str]:
    """A column of alternating colours that never forms a group with its
    neighbours (a checkerboard pattern)."""
    return ["GB"[(row + column) % 2] for row in range(height)]


# --- chain forms ---------------------------------------------------------------


def test_stairs_fires_a_four_chain():
    # 3-1 stairs: each column holds 3 of one colour with the next colour's key
    # puyo on top. A red dropped in column 0 pops the reds, the green key falls
    # next to the greens, and so on to the right.
    rows = [
        ".GBY..",
        ".RGBY.",
        ".RGBY.",
        ".RGBY.",
    ]
    board = Board.from_rows(rows)
    best = best_potential_chain(board)
    assert (best.column, best.color, best.puyos_needed) == (0, "R", 1)
    assert best.chain_length == 4
    assert best.score == 40 + 320 + 640 + 1280
    assert board == Board.from_rows(rows)  # the real board is untouched


def test_sandwich_fires_a_four_chain():
    # In columns 3-5, each link's colour is sandwiched between puyos of the next
    # link's colour (R R Y R, G G R G, B B G B B, read bottom to top). A yellow
    # dropped in column 2 pops the yellows, the red key falls onto the reds and
    # joins the red to their right, and so on to the right.
    board = Board.from_rows([
            ".....B",
            "...RGB",
            "...YRG",
            "..YRGB",
            "..YRGB",
        ])
    chains = find_potential_chains(board)
    # The other triggers need 2 puyos and pop a single group.
    assert [(c.column, c.color, c.chain_length) for c in chains] == [(1, "Y", 1), (2, "Y", 4), (4, "B", 1), (5, "B", 1)]
    best = best_potential_chain(board)
    assert (best.column, best.color, best.puyos_needed) == (2, "Y", 1)
    assert best.chain_length == 4
    assert best.score == 40 + 320 + 640 + 1280


# --- drop rules ----------------------------------------------------------------


def test_two_puyos_needed():
    board = Board.from_rows(["RR...."])
    best = best_potential_chain(board)
    assert best.puyos_needed == 2
    assert best.chain_length == 1
    assert best_potential_chain(board, max_puyos=1) is None


def test_empty_board_has_no_potential_chain():
    assert best_potential_chain(Board()) is None


def test_drop_into_hidden_row_does_not_count():
    # Three reds at the top of column 0. The fourth red would land in the
    # hidden 13th row, where puyos don't pop.
    board = Board([filler(0, 9) + ["R"] * 3, filler(1, 12), [], [], [], []])
    assert best_potential_chain(board) is None
    # The same reds one row lower can be completed.
    board = Board([filler(0, 8) + ["R"] * 3, filler(1, 12), [], [], [], []])
    assert best_potential_chain(board).chain_length == 1


def test_matches_trying_every_colour():
    # find_potential_chains only tries colours touching the landing cells.
    # Check that it finds the same chains as trying every colour.
    rng = random.Random(0)
    for seed in range(30):
        game = Game(seed=seed, max_turns=rng.randrange(10, 60))
        while not game.is_over:
            game.play(rng.choice(game.legal_placements()))
        board = game.board

        expected = set()
        for column in range(WIDTH):
            for color in COLORS:
                for count in (1, 2):
                    if board.height(column) + count > VISIBLE_HEIGHT:
                        break
                    trial = board.copy()
                    trial.columns[column] += [color] * count
                    result = trial.resolve()
                    if result.chain_length:
                        expected.add((column, color, count, result))
                        break
        found = {(c.column, c.color, c.puyos_needed, c.result) for c in find_potential_chains(board)}
        assert found == expected
