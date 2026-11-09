"""Puyo Puyo game engine using Tsu rules, the competitive standard.

Coordinates are (column, row). Column 0 is the leftmost column and row 0 is the
bottom row. Rows 0-11 are the 12 visible rows. Row 12 is the hidden 13th row:
puyos can sit there, but they never connect or pop until they fall back into the
visible rows.

Rules source: Puyo Nexus wiki (Basic rules, Scoring, Tsu rule pages).

Simplifications (for now):
- A pair can move sideways from the spawn column only across columns whose top
  visible cell is empty. Climbing tricks and rotation kicks are not modelled.
- Pairs are drawn uniformly at random. The real Tsu randomizer is not modelled.
- Single player only: no nuisance (garbage) puyos yet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

WIDTH = 6
VISIBLE_HEIGHT = 12
HEIGHT = VISIBLE_HEIGHT + 1  # includes the hidden 13th row
SPAWN_COLUMN = 2  # third column from the left
POP_SIZE = 4
NUM_VISIBLE_PAIRS = 3  # the current pair, plus next and next-next
COLORS = ("R", "G", "B", "Y", "P")
EMPTY = "."

Pair = tuple[str, str]  # (axis colour, child colour)

# Tsu chain power for the 1st, 2nd, 3rd, ... chain step. Steps past the end of
# the table use the last value.
CHAIN_POWER = (
    0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288,
    320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672,
)
# Bonus by number of distinct colours popped in one step.
COLOR_BONUS = {1: 0, 2: 3, 3: 6, 4: 12, 5: 24}
# Bonus by group size, for sizes 4..10. Groups of 11 or more get 10.
GROUP_BONUS = (0, 2, 3, 4, 5, 6, 7)


def group_bonus(size: int) -> int:
    """Group bonus for one popped group. (i.e number of puyos popped at once)

    Args:
        size: number of puyos in the group (4 or more).

    Returns:
        The bonus added to the score bracket: 0 for 4 puyos, up to 10 for 11+.
    """
    if size >= 11:
        return 10
    return GROUP_BONUS[size - POP_SIZE]


def step_score(chain: int, group_sizes: list[int], num_colors: int) -> int:
    """Score for one chain step: 10 x puyos cleared x (chain power + colour
    bonus + group bonus), with the bracket clamped to 1..999.

    Args:
        chain: which step of the chain this is (1 for the first pop).
        group_sizes: size of each group popping in this step, e.g. [4, 5].
        num_colors: number of distinct colours popping in this step.

    Returns:
        The points scored by this step.
    """
    power = CHAIN_POWER[min(chain, len(CHAIN_POWER)) - 1]
    bonus = power + COLOR_BONUS[num_colors] + sum(group_bonus(s) for s in group_sizes)
    return 10 * sum(group_sizes) * min(max(bonus, 1), 999)


@dataclass(frozen=True, order=True)
class Placement:
    """Where to drop a pair.

    column is the axis puyo's column. rotation says where the child puyo sits
    relative to the axis: 0 = above, 1 = right, 2 = below, 3 = left.
    """

    column: int
    rotation: int

    @property
    def child_column(self) -> int:
        """The column the child puyo lands in."""
        if self.rotation == 1:
            return self.column + 1
        if self.rotation == 3:
            return self.column - 1
        return self.column


# The 22 placements that fit inside the board, ignoring column heights.
ALL_PLACEMENTS = tuple(
    Placement(column, rotation)
    for rotation in range(4)
    for column in range(WIDTH)
    if 0 <= Placement(column, rotation).child_column < WIDTH
)


@dataclass(frozen=True)
class ChainStep:
    """One step of a chain: everything that pops at the same moment."""

    chain: int  # 1 for the first pop, 2 for the second, ...
    group_sizes: tuple[int, ...]  # size of each group that popped
    colors: frozenset[str]  # colours that popped
    score: int  # points from this step


@dataclass(frozen=True)
class ChainResult:
    """Everything that popped after one placement. No steps means nothing popped."""

    steps: tuple[ChainStep, ...]
    all_clear: bool  # True if the chain emptied the board

    @property
    def chain_length(self) -> int:
        return len(self.steps)

    @property
    def score(self) -> int:
        return sum(step.score for step in self.steps)


class Board:
    """A 6-wide board stored as columns of colours, bottom to top."""

    __slots__ = ("columns",)

    def __init__(self, columns: list[list[str]] | None = None) -> None:
        """
        Args:
            columns: 6 lists of colour letters, each listed bottom to top,
                e.g. [["R", "G"], [], ...]. None gives an empty board.

        Raises:
            ValueError: if there aren't 6 columns or a column is taller than 13.
        """
        if columns is None:
            self.columns: list[list[str]] = [[] for _ in range(WIDTH)]
            return
        if len(columns) != WIDTH:
            raise ValueError(f"expected {WIDTH} columns, got {len(columns)}")
        if any(len(col) > HEIGHT for col in columns):
            raise ValueError(f"a column is taller than {HEIGHT}")
        self.columns = [list(col) for col in columns]

    @classmethod
    def from_rows(cls, rows: list[str]) -> Board:
        """Build a board by drawing it as text, the way it looks on screen.

        The board stores puyos as columns listed bottom to top, which is hard
        to write by hand. This lets you draw the picture instead, which is
        handy for tests and for setting up positions.

        Args:
            rows: up to 13 strings of 6 characters, top row first, with '.' for
                empty. Rows are bottom-aligned: the last string is the bottom
                row, and any rows you leave out above are empty.

        Returns:
            The board.

        Example:
            Board.from_rows([
                "G.....",
                "RRB..Y",
            ])
            gives a board whose columns are
            [["R", "G"], ["R"], ["B"], [], [], ["Y"]]
            i.e. column 0 has R at the bottom with G on top, column 1 has R,
            column 2 has B, and column 5 has Y.

        Raises:
            ValueError: on a wrong row width, too many rows, or a puyo with an
                empty cell below it.
        """
        if len(rows) > HEIGHT:
            raise ValueError(f"at most {HEIGHT} rows, got {len(rows)}")
        columns: list[list[str]] = [[] for _ in range(WIDTH)]
        for row_index, row in enumerate(reversed(rows)):
            if len(row) != WIDTH:
                raise ValueError(f"row {row!r} is not {WIDTH} wide")
            for column, cell in enumerate(row):
                if cell == EMPTY:
                    continue
                if len(columns[column]) != row_index:
                    raise ValueError(f"floating puyo at column {column}, row {row_index}")
                columns[column].append(cell)
        return cls(columns)

    def to_rows(self) -> list[str]:
        """Draw the board as text, the reverse of from_rows.

        Used by print(board), and handy for checking or saving a position.
        Passing the output back to from_rows gives the same board.

        Returns:
            13 strings of 6 characters, one per row, top (hidden) row first,
            with '.' for empty.

        Example:
            For the board from the from_rows example, to_rows() returns
            [
                "......",  # row 13 (hidden)
                "......",
                "......",
                "......",
                "......",
                "......",
                "......",
                "......",
                "......",
                "......",
                "......",
                "G.....",
                "RRB..Y",  # row 1 (bottom)
            ]
        """
        return [
            "".join(self.get(column, row) for column in range(WIDTH))
            for row in reversed(range(HEIGHT))
        ]

    def copy(self) -> Board:
        board = Board.__new__(Board)
        board.columns = [col[:] for col in self.columns]
        return board

    def height(self, column: int) -> int:
        """Number of puyos in the column (0-13)."""
        return len(self.columns[column])

    def get(self, column: int, row: int) -> str:
        """The colour letter at (column, row), or '.' if the cell is empty."""
        col = self.columns[column]
        return col[row] if row < len(col) else EMPTY

    def is_empty(self) -> bool:
        return not any(self.columns)

    def is_dead(self) -> bool:
        """True when the top visible cell of the spawn column is filled."""
        return self.height(SPAWN_COLUMN) >= VISIBLE_HEIGHT

    def is_reachable(self, placement: Placement) -> bool:
        """Whether a pair can be moved from the spawn column to this placement.

        Every column between the spawn column and the target columns must
        have its top visible cell empty.
        """
        if placement not in ALL_PLACEMENTS or self.is_dead():
            return False
        low = min(SPAWN_COLUMN, placement.column, placement.child_column)
        high = max(SPAWN_COLUMN, placement.column, placement.child_column)
        return all(self.height(c) < VISIBLE_HEIGHT for c in range(low, high + 1))

    def legal_placements(self, pair: Pair) -> list[Placement]:
        """Where this pair can be dropped.

        Args:
            pair: (axis colour, child colour), e.g. ("R", "G").

        Returns:
            Reachable placements: up to 22, or up to 11 for a same-colour pair
            because its mirror-image placements are dropped as duplicates.
            Empty if the board is dead.
        """
        same_color = pair[0] == pair[1]
        return [
            p
            for p in ALL_PLACEMENTS
            # For a same-colour pair, rotation 2 repeats rotation 0 in the same
            # column, and rotation 3 repeats rotation 1 one column to the left.
            if not (same_color and p.rotation >= 2) and self.is_reachable(p)
        ]

    def place(self, pair: Pair, placement: Placement) -> None:
        """Drop a pair onto the board, changing it in place. Chains are not
        resolved; call resolve() for that.

        Horizontal pairs split and each puyo falls on its own column. Does not
        check reachability; use is_reachable() or legal_placements() first.

        Args:
            pair: (axis colour, child colour).
            placement: where to drop it.

        Raises:
            ValueError: if a column would go above 13 puyos.
        """
        axis, child = pair
        column = self.columns[placement.column]
        if placement.rotation == 0:
            column += [axis, child]
        elif placement.rotation == 2:
            column += [child, axis]
        else:
            column.append(axis)
            self.columns[placement.child_column].append(child)
        if len(column) > HEIGHT or len(self.columns[placement.child_column]) > HEIGHT:
            raise ValueError(f"{placement} overflows the board")

    def find_groups(self) -> list[list[tuple[int, int]]]:
        """Find the groups that would pop right now.

        Returns:
            One list per group of 4 or more connected same-colour puyos in the
            visible rows, each holding the group's (column, row) cells.
            Empty if nothing would pop.
        """
        seen: set[tuple[int, int]] = set()
        groups = []
        for column in range(WIDTH):
            for row in range(min(self.height(column), VISIBLE_HEIGHT)):
                if (column, row) in seen:
                    continue
                color = self.columns[column][row]
                seen.add((column, row))
                group = []
                stack = [(column, row)]
                while stack:
                    c, r = stack.pop()
                    group.append((c, r))
                    for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                        if (
                            0 <= nc < WIDTH
                            and 0 <= nr < min(self.height(nc), VISIBLE_HEIGHT)
                            and (nc, nr) not in seen
                            and self.columns[nc][nr] == color
                        ):
                            seen.add((nc, nr))
                            stack.append((nc, nr))
                if len(group) >= POP_SIZE:
                    groups.append(group)
        return groups

    def resolve(self) -> ChainResult:
        """Pop groups and apply gravity until nothing pops, changing the board
        in place.

        Returns:
            The chain that happened, with each step's score.
        """
        steps = []
        while groups := self.find_groups():
            sizes = [len(group) for group in groups]
            colors = frozenset(self.columns[g[0][0]][g[0][1]] for g in groups)
            chain = len(steps) + 1
            steps.append(ChainStep(chain, tuple(sizes), colors, step_score(chain, sizes, len(colors))))

            popped: list[set[int]] = [set() for _ in range(WIDTH)]
            for group in groups:
                for c, r in group:
                    popped[c].add(r)
            for c in range(WIDTH):
                if popped[c]:
                    self.columns[c] = [p for r, p in enumerate(self.columns[c]) if r not in popped[c]]
        return ChainResult(tuple(steps), all_clear=bool(steps) and self.is_empty())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.columns == other.columns

    def __str__(self) -> str:
        rows = self.to_rows()
        return "\n".join([rows[0] + "  (hidden row 13)"] + rows[1:])

    def __repr__(self) -> str:
        return f"Board.from_rows({[r for r in self.to_rows() if r != EMPTY * WIDTH]!r})"


class PairSource:
    """A reproducible, endless stream of random pairs.

    Two games with the same seed get the same pairs, so bots can be compared
    on identical piece sequences.
    """

    def __init__(self, seed: int | None = None, num_colors: int = 4) -> None:
        """
        Args:
            seed: random seed. None gives a different sequence every time.
            num_colors: how many colours pairs are drawn from (3-5).
        """
        if not 3 <= num_colors <= len(COLORS):
            raise ValueError(f"num_colors must be 3..{len(COLORS)}")
        self.colors = COLORS[:num_colors]
        self.rng = random.Random(seed)
        self.pairs: list[Pair] = []

    def __getitem__(self, index: int) -> Pair:
        """The pair for turn `index` (0 is the first pair of the game), as
        (axis colour, child colour). Pairs are generated on first use."""
        while len(self.pairs) <= index:
            self.pairs.append((self.rng.choice(self.colors), self.rng.choice(self.colors)))
        return self.pairs[index]


class Game:
    """A single-player game. Only the current pair and the next two are visible.

    Bots should read pieces through visible_pairs(). pair_source holds the
    whole future sequence, and reading ahead in it would be cheating.
    """

    def __init__(self, seed: int | None = None, num_colors: int = 4, max_turns: int | None = None) -> None:
        """
        Args:
            seed: seed for the pair sequence. Games with the same seed get the
                same pairs. None gives a different sequence every time.
            num_colors: how many colours pairs are drawn from (3-5).
            max_turns: end the game after this many placements. None means
                play until the board tops out.
        """
        self.board = Board()
        self.max_turns = max_turns
        self.turn = 0
        self.score = 0
        self.pair_source = PairSource(seed, num_colors)

    @property
    def current_pair(self) -> Pair:
        return self.pair_source[self.turn]

    def visible_pairs(self) -> list[Pair]:
        """[current pair, next pair, next-next pair], each as
        (axis colour, child colour)."""
        return [self.pair_source[self.turn + i] for i in range(NUM_VISIBLE_PAIRS)]

    @property
    def is_over(self) -> bool:
        """True once the board has topped out or max_turns placements are done."""
        out_of_turns = self.max_turns is not None and self.turn >= self.max_turns
        return out_of_turns or self.board.is_dead()

    def legal_placements(self) -> list[Placement]:
        """Where the current pair can be dropped. Empty once the game is over."""
        return [] if self.is_over else self.board.legal_placements(self.current_pair)

    def play(self, placement: Placement) -> ChainResult:
        """Drop the current pair, resolve any chain, add its score and move on
        to the next turn.

        Args:
            placement: where to drop the current pair.

        Returns:
            The chain that the placement set off (no steps if nothing popped).

        Raises:
            RuntimeError: if the game is already over.
            ValueError: if the placement isn't reachable.
        """
        if self.is_over:
            raise RuntimeError("the game is over")
        if not self.board.is_reachable(placement):
            raise ValueError(f"{placement} is not reachable")
        self.board.place(self.current_pair, placement)
        result = self.board.resolve()
        self.score += result.score
        self.turn += 1
        return result
