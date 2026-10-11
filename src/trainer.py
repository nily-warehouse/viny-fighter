from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import neat

from .config import HarnessConfig
from .actions import enabled_actions
from .configs import config_dir
from .evaluator import EvaluationCallback, EvaluationResult, HarnessEvaluator
from .neat_model import genome_to_model, load_config


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    species: int
    transitions: int
    inferred_kills: int
    failed_evaluations: int
    best_model: str


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Trainer:
    def __init__(self, harness: HarnessConfig, population: neat.Population) -> None:
        self.harness = harness
        self.population = population
        self.evaluator = HarnessEvaluator(harness)
        self.history_path = harness.project_root / "pool" / "training_history.jsonl"
        self._stats: GenerationStats | None = None
        self._results: list[EvaluationResult] = []
        self._on_evaluation: EvaluationCallback | None = None

    @classmethod
    def create(cls, harness: HarnessConfig, population_size: int) -> "Trainer":
        actions_file = config_dir(harness.config_id) / "actions.json"
        config = load_config(
            str(harness.neat_config), population_size,
            outputs=enabled_actions(actions_file), actions_file=str(actions_file),
        )
        config.config_id = harness.config_id
        return cls(harness, neat.Population(config))

    @classmethod
    def restore(cls, harness: HarnessConfig, generation: int) -> "Trainer":
        if generation < 0:
            raise ValueError("generation must be zero or greater")
        checkpoint = (
            harness.generations_dir
            / f"gen-{generation}"
            / f"checkpoint-{generation}"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"generation {generation} does not exist: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(str(checkpoint))
        actions_file = config_dir(harness.config_id) / "actions.json"
        selected_config = load_config(
            str(harness.neat_config),
            outputs=enabled_actions(actions_file), actions_file=str(actions_file),
        )
        saved_id = getattr(population.config, "config_id", 1)
        if saved_id != harness.config_id:
            raise ValueError(
                f"checkpoint uses config {saved_id}, requested config {harness.config_id}"
            )
        saved_actions = getattr(population.config, "action_names", selected_config.action_names)
        if (saved_actions != selected_config.action_names
            or population.config.genome_config.num_outputs != len(selected_config.action_names)):
            raise ValueError("selected actions do not match the checkpoint")
        population.config.action_names = selected_config.action_names
        population.config.config_id = harness.config_id
        if population.generation != generation:
            raise ValueError(
                f"checkpoint contains generation {population.generation}, "
                f"expected {generation}"
            )
        return cls(harness, population)

    def generation_dir(self, generation: int) -> Path:
        return self.harness.generations_dir / f"gen-{generation}"

    def checkpoint_path(self, generation: int) -> Path:
        return self.generation_dir(generation) / f"checkpoint-{generation}"

    def _save_checkpoint(self) -> None:
        generation = self.population.generation
        directory = self.generation_dir(generation)
        directory.mkdir(parents=True, exist_ok=True)
        checkpointer = neat.Checkpointer(
            generation_interval=1,
            filename_prefix=str(directory / "checkpoint-"),
        )
        # neat-python prints directly to stdout here. The CLI owns presentation,
        # so keep the library call quiet and report the saved generation there.
        with contextlib.redirect_stdout(io.StringIO()):
            checkpointer.save_checkpoint(
                self.population.config,
                self.population.population,
                self.population.species,
                generation,
            )

    @staticmethod
    def _clear_brains(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("brain_*.json"):
            path.unlink()

    def _write_models(
        self,
        directory: Path,
        genomes: list[tuple[int, neat.DefaultGenome]],
        generation: int,
    ) -> list[str]:
        self._clear_brains(directory)
        names: list[str] = []
        for index, (_, genome) in enumerate(genomes, start=1):
            name = f"brain_{index}.json"
            _atomic_json(
                directory / name,
                genome_to_model(genome, self.population.config, generation),
            )
            names.append(name)
        return names

    def export_population(
        self, genomes: list[tuple[int, neat.DefaultGenome]] | None = None
    ) -> list[str]:
        generation = self.population.generation
        ordered = (
            sorted(self.population.population.items())
            if genomes is None
            else genomes
        )
        names = self._write_models(self.harness.models_dir, ordered, generation)
        self._write_models(
            self.generation_dir(generation) / "models", ordered, generation
        )
        return names

    def initialize(self) -> list[str]:
        self._save_checkpoint()
        return self.export_population()

    def _record_stats(self, stats: GenerationStats) -> None:
        value = asdict(stats)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        _atomic_json(self.generation_dir(stats.generation) / "stats.json", value)

    def _save_best(self, genome: neat.DefaultGenome, generation: int) -> None:
        path = self.harness.models_dir / "best_brain.json"
        previous = -math.inf
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as stream:
                    previous = float(json.load(stream).get("fitness", -math.inf))
            except (OSError, ValueError, TypeError):
                previous = -math.inf
        if genome.fitness is not None and genome.fitness > previous:
            _atomic_json(path, genome_to_model(genome, self.population.config, generation))

    def _evaluate_generation(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        _: neat.Config,
    ) -> None:
        generation = self.population.generation
        ordered = sorted(genomes)
        names = self.export_population(ordered)
        results = asyncio.run(
            self.evaluator.evaluate(names, on_result=self._on_evaluation)
        )
        failures = [result for result in results if result.failed]
        if failures:
            names = ", ".join(result.model_name for result in failures[:3])
            raise RuntimeError(
                f"{len(failures)} evaluations failed or produced invalid data ({names}); "
                "the generation was not advanced"
            )

        for result in results:
            ordered[result.index][1].fitness = result.report.fitness

        champion_index, (_, champion) = max(
            enumerate(ordered), key=lambda item: float(item[1][1].fitness)
        )
        fitnesses = [float(genome.fitness) for _, genome in ordered]
        stats = GenerationStats(
            generation=generation,
            best_fitness=float(champion.fitness),
            mean_fitness=sum(fitnesses) / len(fitnesses),
            species=len(self.population.species.species),
            transitions=sum(result.report.transitions for result in results),
            inferred_kills=sum(result.report.inferred_kills for result in results),
            failed_evaluations=sum(result.failed for result in results),
            best_model=names[champion_index],
        )
        self.export_population(ordered)
        self._save_best(champion, generation)
        self._record_stats(stats)
        self._stats = stats
        self._results = results

    def advance(
        self,
        generations: int,
        on_generation: Callable[[GenerationStats, list[EvaluationResult]], None]
        | None = None,
        on_generation_start: Callable[[int, int], None] | None = None,
        on_evaluation: EvaluationCallback | None = None,
    ) -> list[GenerationStats]:
        if generations < 1:
            raise ValueError("generations must be at least one")
        history: list[GenerationStats] = []

        for _ in range(generations):
            target = self.population.generation + 1
            if self.checkpoint_path(target).exists():
                raise FileExistsError(
                    f"generation {target} already exists; continue from that generation"
                )
            self._stats = None
            self._results = []
            self._on_evaluation = on_evaluation
            if on_generation_start is not None:
                on_generation_start(
                    self.population.generation,
                    len(self.population.population),
                )
            try:
                self.population.run(self._evaluate_generation, 1)
            finally:
                self._on_evaluation = None
            if self._stats is None:
                raise RuntimeError("generation finished without evaluation statistics")
            self._save_checkpoint()
            self.export_population()
            history.append(self._stats)
            if on_generation is not None:
                on_generation(self._stats, self._results)

        return history
