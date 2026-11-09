"""Bots that decide where to drop each pair.

A bot only sees what a player sees: the board and the visible pairs (current,
next, next-next). It never gets the Game, so it can't read future pairs.
"""

from __future__ import annotations

import random
from typing import Protocol

from puyo.engine import Board, Pair, Placement


class Bot(Protocol):
    name: str

    def choose(self, board: Board, pairs: list[Pair]) -> Placement:
        """Pick where to drop the current pair.

        Args:
            board: the current board. Copy it before trying moves on it.
            pairs: the visible pairs, [current, next, next-next], each as
                (axis colour, child colour).

        Returns:
            A placement for pairs[0], one of board.legal_placements(pairs[0]).
        """
        ...


class RandomBot:
    """Drops each pair in a random legal spot."""

    name = "random"

    def __init__(self, seed: int | str | None = None) -> None:
        """
        Args:
            seed: seed for the bot's own random choices. Keep it different
                from the game's seed so the two random streams don't line up.
        """
        self.rng = random.Random(seed)

    def choose(self, board: Board, pairs: list[Pair]) -> Placement:
        return self.rng.choice(board.legal_placements(pairs[0]))


class GreedyBot:
    """Takes whichever placement scores the most points right now.

    It looks only at the current pair. It avoids placements that top out the
    board when there's another option, and breaks ties at random. It has no
    idea of building chains, so it pops small groups as soon as it can.
    """

    name = "greedy"

    def __init__(self, seed: int | str | None = None) -> None:
        """
        Args:
            seed: seed for breaking ties between equally good placements.
        """
        self.rng = random.Random(seed)

    def choose(self, board: Board, pairs: list[Pair]) -> Placement:
        scored = []
        for placement in board.legal_placements(pairs[0]):
            after = board.copy()
            after.place(pairs[0], placement)
            score = after.resolve().score
            # A move that tops out is only taken if every move does.
            scored.append((-1 if after.is_dead() else score, placement))
        best = max(score for score, placement in scored)
        return self.rng.choice([placement for score, placement in scored if score == best])
