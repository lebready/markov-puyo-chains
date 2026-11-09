"""Potential-chain detection: find the chains a board could fire.

Strong Puyo AIs judge a board by the chain it could set off next, not just by
the points it has already scored. To find those chains, drop a few imaginary
puyos of one colour on top of a column and see what pops. Trying every column
and colour gives the board's potential chains.

Simplification: drops ignore whether a pair could actually reach the column
(see Board.is_reachable).
"""

from __future__ import annotations

from dataclasses import dataclass

from puyo.engine import EMPTY, VISIBLE_HEIGHT, WIDTH, Board, ChainResult


@dataclass(frozen=True)
class PotentialChain:
    """A chain the board would fire if some puyos were dropped on it."""

    column: int  # where the extra puyos are dropped
    color: str  # colour of the extra puyos
    puyos_needed: int  # fewest extra puyos that set the chain off
    result: ChainResult  # the chain that fires

    @property
    def chain_length(self) -> int:
        return self.result.chain_length

    @property
    def score(self) -> int:
        return self.result.score


def touching_colors(board: Board, column: int, count: int) -> set[str]:
    """Colours that could pop if `count` puyos were dropped on this column.

    A dropped puyo only pops if it touches a puyo of its own colour, so these
    are the colours next to the cells the drop would fill.

    Args:
        board: the board to drop onto.
        column: the column to drop on.
        count: how many puyos would be dropped.

    Returns:
        Colours found in the visible cells below, left of or right of the
        landing cells.
    """
    bottom = board.height(column)
    cells = [(column, bottom - 1)]
    for row in range(bottom, bottom + count):
        cells += [(column - 1, row), (column + 1, row)]
    colors = {board.get(c, r) for c, r in cells if 0 <= c < WIDTH and 0 <= r < VISIBLE_HEIGHT}
    return colors - {EMPTY}


def find_potential_chains(board: Board, max_puyos: int = 2) -> list[PotentialChain]:
    """Every chain the board could fire by dropping puyos of one colour on one
    column.

    For each column and colour, drops 1 puyo, then 2, and so on up to
    max_puyos, stopping at the first count that sets off a chain. A drop that
    would put a puyo in the hidden 13th row is skipped, because puyos there
    can't pop.

    Args:
        board: a settled board, with nothing waiting to pop (e.g. after
            resolve()). It is not changed.
        max_puyos: the most puyos to drop at once.

    Returns:
        At most one chain per (column, colour), with the fewest puyos that
        fire it. Ordered by column, then colour. Empty if nothing can fire.
    """
    chains = []
    for column in range(WIDTH):
        most = min(max_puyos, VISIBLE_HEIGHT - board.height(column))
        for color in sorted(touching_colors(board, column, most)):
            for count in range(1, most + 1):
                trial = board.copy()
                trial.columns[column] += [color] * count
                result = trial.resolve()
                if result.chain_length:
                    chains.append(PotentialChain(column, color, count, result))
                    break
    return chains


def best_potential_chain(board: Board, max_puyos: int = 2) -> PotentialChain | None:
    """The biggest chain the board could fire.

    The longest chain wins. Ties go to the higher score, then to fewer puyos
    needed, then to the leftmost column.

    Args:
        board: a settled board, with nothing waiting to pop. It is not changed.
        max_puyos: the most puyos to drop at once.

    Returns:
        The best potential chain, or None if no drop sets anything off.
    """
    return max(
        find_potential_chains(board, max_puyos),
        key=lambda chain: (chain.chain_length, chain.score, -chain.puyos_needed),
        default=None,
    )
