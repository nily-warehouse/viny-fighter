from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SERVER_SLOTS, HarnessConfig, PROJECT_ROOT
from src.evaluator import EvaluationResult
from src.trainer import GenerationStats, Trainer


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _population_size(value: str) -> int:
    number = _positive_int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return number


def _generation(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Train bot03 policies with NEAT.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    initialize = commands.add_parser(
        "init",
        help="create generation 0",
        description="Create a new generation 0 population.",
    )
    initialize.add_argument(
        "--population",
        type=_population_size,
        default=50,
        metavar="SIZE",
        help="number of brains to create (default: 50)",
    )

    train = commands.add_parser(
        "train",
        help="continue a saved generation",
        description="Evaluate and evolve a saved population.",
    )
    train.add_argument(
        "--from-generation",
        type=_generation,
        required=True,
        metavar="N",
        help="saved generation to resume",
    )
    train.add_argument(
        "--generations",
        type=_positive_int,
        default=1,
        metavar="COUNT",
        help="number of generations to run (default: 1)",
    )
    train.add_argument(
        "--episodes",
        type=_positive_int,
        default=3,
        metavar="COUNT",
        help="episodes per brain (default: 3)",
    )
    for command in (initialize, train):
        command.add_argument(
            "--config",
            type=_positive_int,
            default=1,
            metavar="ID",
            help="configuration directory under src/configs (default: 1)",
        )
    return parser


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


class TrainingConsole:
    """Small terminal renderer; training logic stays independent from presentation."""

    COLORS = {
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "dim": "\033[2m",
        "bold": "\033[1m",
    }
    RESET = "\033[0m"

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.color = self.stream.isatty() and "NO_COLOR" not in os.environ
        self._generation_started = 0.0

    def _style(self, value: object, color: str) -> str:
        text = str(value)
        if not self.color:
            return text
        return f"{self.COLORS[color]}{text}{self.RESET}"

    def _write(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def heading(self, title: str) -> None:
        self._write(self._style(title, "bold"))
        self._write(self._style("─" * len(title), "dim"))

    def field(self, label: str, value: object) -> None:
        self._write(f"  {self._style(f'{label:<12}', 'dim')}{value}")

    def init_started(self, population: int) -> None:
        self.heading("Minebot · initialize")
        self.field("Population", f"{population} brains")
        self.field("Generation", "0")
        self._write()
        self._write(f"{self._style('●', 'cyan')} Creating population…")

    def init_finished(self, trainer: Trainer, model_count: int) -> None:
        self._write(
            f"{self._style('✓', 'green')} Generation 0 is ready "
            f"({model_count} models)"
        )
        self.field("Checkpoint", trainer.checkpoint_path(0))
        self.field("Models", trainer.harness.models_dir)

    def train_started(
        self,
        trainer: Trainer,
        from_generation: int,
        generations: int,
        episodes: int,
    ) -> None:
        population = len(trainer.population.population)
        slots = min(SERVER_SLOTS, population)
        self.heading("Minebot · train")
        self.field(
            "Run",
            f"generation {from_generation} → {from_generation + generations}",
        )
        self.field("Population", f"{population} brains")
        self.field("Episodes", f"{episodes} per brain")
        self.field("Workers", f"{slots} Minecraft slots")
        self.field("Output", trainer.harness.generations_dir)

    def generation_started(self, generation: int, population: int) -> None:
        self._generation_started = time.monotonic()
        self._write()
        self._write(
            f"{self._style(f'Generation {generation}', 'bold')} "
            f"{self._style(f'· evaluating {population} brains', 'dim')}"
        )

    def evaluation_finished(
        self,
        result: EvaluationResult,
        completed: int,
        total: int,
    ) -> None:
        succeeded = not result.failed
        symbol = self._style(
            "✓" if succeeded else "!",
            "green" if succeeded else "red",
        )
        progress = self._style(f"[{completed:>{len(str(total))}}/{total}]", "dim")
        score = f"{result.report.fitness:.3f}" if succeeded else "invalid"
        details = (
            f"fitness {score:>8}  "
            f"steps {result.report.transitions:5d}  "
            f"kills {result.report.inferred_kills:2d}  "
            f"slot {result.slot:2d}  {_duration(result.elapsed_seconds):>7}"
        )
        self._write(f"  {progress} {symbol} {result.model_name:<20} {details}")

        if result.report.error:
            self._write(self._style(f"          │ {result.report.error}", "red"))
        if not succeeded and result.output_tail.strip():
            tail = result.output_tail.strip().splitlines()[-3:]
            for line in tail:
                self._write(self._style(f"          │ {line}", "red"))

    def generation_finished(
        self,
        stats: GenerationStats,
        results: list[EvaluationResult],
    ) -> None:
        elapsed = time.monotonic() - self._generation_started
        successful = len(results) - stats.failed_evaluations
        self._write(
            f"{self._style('✓', 'green')} Generation {stats.generation} complete "
            f"{self._style(f'· {_duration(elapsed)}', 'dim')}"
        )
        self.field(
            "Fitness",
            f"best {stats.best_fitness:.3f} · mean {stats.mean_fitness:.3f}",
        )
        self.field(
            "Results",
            f"{successful}/{len(results)} succeeded · "
            f"{stats.transitions} transitions · {stats.inferred_kills} kills",
        )
        self.field("Species", stats.species)
        self.field("Champion", stats.best_model)
        self.field("Checkpoint", f"generation {stats.generation + 1}")

    def train_finished(self, generation: int) -> None:
        self._write()
        message = f"Training finished at generation {generation}"
        self._write(f"{self._style('✓', 'green')} {message}")

    def interrupted(self) -> None:
        self._write()
        self._write(f"{self._style('!', 'yellow')} Training interrupted")

    def error(self, message: str) -> None:
        self._write(f"{self._style('✗', 'red')} {message}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    console = TrainingConsole()

    try:
        if args.command == "init":
            console.init_started(args.population)
            harness = HarnessConfig(project_root=PROJECT_ROOT, config_id=args.config)
            trainer = Trainer.create(harness, args.population)
            names = trainer.initialize()
            console.init_finished(trainer, len(names))
            return

        harness = HarnessConfig(
            project_root=PROJECT_ROOT, episodes=args.episodes, config_id=args.config
        )
        trainer = Trainer.restore(harness, args.from_generation)
        console.train_started(
            trainer,
            args.from_generation,
            args.generations,
            args.episodes,
        )
        trainer.advance(
            args.generations,
            on_generation=console.generation_finished,
            on_generation_start=console.generation_started,
            on_evaluation=console.evaluation_finished,
        )
        console.train_finished(trainer.population.generation)
    except KeyboardInterrupt:
        console.interrupted()
        raise SystemExit(130) from None
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        console.error(str(error))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
