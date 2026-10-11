from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, replace
from typing import Callable

from .config import SERVER_SLOTS, HarnessConfig
from .configs import fitness_module
from .fitness_report import FitnessReport
from .transitions import clear_transition_file, consume_transition_file


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    model_name: str


@dataclass(frozen=True)
class EvaluationResult:
    index: int
    model_name: str
    slot: int
    report: FitnessReport
    return_code: int | None
    output_tail: str
    elapsed_seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return (
            self.return_code != 0
            or self.report.transitions == 0
            or self.report.error is not None
            or not math.isfinite(self.report.fitness)
        )


EvaluationCallback = Callable[[EvaluationResult, int, int], None]


class HarnessEvaluator:
    """Run bot03 processes concurrently, with one worker per server slot."""

    def __init__(self, harness: HarnessConfig) -> None:
        if harness.episodes < 1:
            raise ValueError("episodes must be at least 1")
        self.harness = harness
        self.fitness_module = fitness_module(harness.config_id)
        self.reward_config = self.fitness_module.RewardConfig()

    async def evaluate_one(self, job: EvaluationJob, slot: int) -> EvaluationResult:
        started_at = time.monotonic()
        transition_path = self.harness.transition_path(slot)
        clear_transition_file(transition_path)
        try:
            process = await asyncio.create_subprocess_exec(
                "node",
                str(self.harness.client_path),
                "instant-run",
                f"ep:{self.harness.episodes}",
                f"policy:{job.model_name}",
                f"slot:{slot}",
                f"config:{self.harness.config_id}",
                cwd=str(self.harness.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError:
            consume_transition_file(transition_path)
            raise

        timeout = self.harness.episodes * self.harness.episode_timeout_seconds + 60
        timed_out = False
        
        try:
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                process.kill()
                output, _ = await process.communicate()
        
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            raise
        
        finally:
            transitions, invalid = consume_transition_file(transition_path)

        report = self.fitness_module.evaluate_fitness(
            transitions,
            expected_episodes=self.harness.episodes,
            config=self.reward_config,
        )
        if invalid:
            report = replace(report, error=f"{invalid} invalid transition records")
        if timed_out:
            report = replace(report, error="evaluation process timed out")
        text = output.decode("utf-8", errors="replace")
        return EvaluationResult(
            index=job.index,
            model_name=job.model_name,
            slot=slot,
            report=report,
            return_code=process.returncode,
            output_tail=text[-3000:],
            elapsed_seconds=time.monotonic() - started_at,
        )

    async def evaluate(
        self,
        model_names: list[str],
        on_result: EvaluationCallback | None = None,
    ) -> list[EvaluationResult]:
        if not self.harness.client_path.is_file():
            raise FileNotFoundError(f"bot03 client not found: {self.harness.client_path}")
        queue: asyncio.Queue[EvaluationJob] = asyncio.Queue()
        for index, model_name in enumerate(model_names):
            queue.put_nowait(EvaluationJob(index=index, model_name=model_name))
        results: list[EvaluationResult | None] = [None] * len(model_names)
        completed = 0

        async def worker(slot: int) -> None:
            nonlocal completed
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    result = await self.evaluate_one(job, slot)
                    results[job.index] = result
                    completed += 1
                    if on_result is not None:
                        on_result(result, completed, len(model_names))
                finally:
                    queue.task_done()

        worker_count = min(SERVER_SLOTS, len(model_names))
        await asyncio.gather(*(worker(slot + 1) for slot in range(worker_count)))
        if any(result is None for result in results):
            raise RuntimeError("an evaluation worker exited without a result")
        return [result for result in results if result is not None]
