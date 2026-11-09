"""Compare bots by playing each one on the same piece sequences.

Run it with:
    uv run python -m puyo.evaluate --games 100 --max-turns 100

Game i uses seed i for its pairs, so every bot sees exactly the same pieces.
That makes the comparison fairer: differences come from the bots, not from
luckier pieces.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from puyo.bots import Bot, GreedyBot, RandomBot
from puyo.engine import Game

# Bot name -> function that builds the bot from a seed.
BOTS: dict[str, Callable[[str], Bot]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
}


@dataclass(frozen=True)
class GameRecord:
    """The outcome of one game."""

    seed: int  # seed of the pair sequence
    score: int
    turns: int  # placements made before the game ended
    max_chain: int  # longest chain in the game
    topped_out: bool  # False if the game ended by reaching max_turns
    think_seconds: float  # total time the bot spent choosing moves


@dataclass(frozen=True)
class Summary:
    """One bot's results over many games."""

    bot: str
    games: int
    mean_score: float
    score_ci: tuple[float, float]  # 95% bootstrap confidence interval for mean_score
    median_score: float
    mean_max_chain: float
    best_chain: int
    top_out_rate: float  # share of games that ended by topping out
    mean_turns: float
    ms_per_move: float


def play_game(bot: Bot, seed: int, max_turns: int, num_colors: int = 4) -> GameRecord:
    """Play one game from an empty board.

    Args:
        bot: the bot making the moves.
        seed: seed for the pair sequence. The same seed gives the same pairs.
        max_turns: stop after this many placements if the board hasn't
            topped out.
        num_colors: how many colours pairs are drawn from (3-5).

    Returns:
        The game's score, length, longest chain and timing.
    """
    game = Game(seed=seed, num_colors=num_colors, max_turns=max_turns)
    max_chain = 0
    think_seconds = 0.0
    while not game.is_over:
        board = game.board.copy()  # so a bot can't change the real board
        start = time.perf_counter()
        placement = bot.choose(board, game.visible_pairs())
        think_seconds += time.perf_counter() - start
        result = game.play(placement)
        max_chain = max(max_chain, result.chain_length)
    return GameRecord(seed, game.score, game.turn, max_chain, game.board.is_dead(), think_seconds)


def bootstrap_mean_ci(
    values: list[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Args:
        values: one number per game, e.g. each game's score.
        confidence: coverage of the interval, e.g. 0.95.
        resamples: how many bootstrap resamples to draw.
        seed: seed for the resampling, so the interval is reproducible.

    Returns:
        (low, high) bounds of the interval.
    """
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


def summarize(bot_name: str, records: list[GameRecord]) -> Summary:
    """Turn one bot's game records into summary statistics.

    Args:
        bot_name: name shown in the results table.
        records: one record per game, from play_game.

    Returns:
        Averages, medians and a confidence interval over the games.
    """
    scores = [r.score for r in records]
    total_turns = sum(r.turns for r in records)
    return Summary(
        bot=bot_name,
        games=len(records),
        mean_score=statistics.fmean(scores),
        score_ci=bootstrap_mean_ci(scores),
        median_score=statistics.median(scores),
        mean_max_chain=statistics.fmean(r.max_chain for r in records),
        best_chain=max(r.max_chain for r in records),
        top_out_rate=sum(r.topped_out for r in records) / len(records),
        mean_turns=total_turns / len(records),
        ms_per_move=1000 * sum(r.think_seconds for r in records) / max(total_turns, 1),
    )


def run(bot_names: list[str], games: int, max_turns: int, first_seed: int = 0, num_colors: int = 4) -> list[Summary]:
    """Play every bot on the same games.

    Args:
        bot_names: keys of BOTS, e.g. ["random", "greedy"].
        games: number of games per bot.
        max_turns: placement limit per game.
        first_seed: game i uses seed first_seed + i.
        num_colors: how many colours pairs are drawn from (3-5).

    Returns:
        One summary per bot, in the order given.
    """
    seeds = range(first_seed, first_seed + games)
    summaries = []
    for name in bot_names:
        # The bot gets its own seed, separate from the pair seed, so its
        # random choices aren't tied to the pieces.
        records = [play_game(BOTS[name](f"{name}-{seed}"), seed, max_turns, num_colors) for seed in seeds]
        summaries.append(summarize(name, records))
    return summaries


def format_table(summaries: list[Summary]) -> str:
    """Lay out summaries as a plain-text table, one row per bot."""
    header = (
        f"{'bot':<8}{'games':>6}{'mean score':>12}{'95% CI':>20}{'median':>9}"
        f"{'mean max chain':>16}{'best chain':>12}{'topped out':>12}{'turns':>7}{'ms/move':>9}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        ci = f"{s.score_ci[0]:,.0f}-{s.score_ci[1]:,.0f}"
        lines.append(
            f"{s.bot:<8}{s.games:>6}{s.mean_score:>12,.0f}{ci:>20}{s.median_score:>9,.0f}"
            f"{s.mean_max_chain:>16.2f}{s.best_chain:>12}{s.top_out_rate:>12.0%}{s.mean_turns:>7.1f}"
            f"{s.ms_per_move:>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Puyo bots on the same piece sequences.")
    parser.add_argument("--bots", nargs="+", choices=sorted(BOTS), default=list(BOTS))
    parser.add_argument("--games", type=int, default=100, help="games per bot")
    parser.add_argument("--max-turns", type=int, default=100, help="placement limit per game")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--colors", type=int, default=4, help="number of colours (3-5)")
    args = parser.parse_args()
    summaries = run(args.bots, args.games, args.max_turns, args.first_seed, args.colors)
    print(format_table(summaries))


if __name__ == "__main__":
    main()
