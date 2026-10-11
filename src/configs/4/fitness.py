from __future__ import annotations

import math
from typing import Any, Iterable

from ...fitness_report import FitnessReport
from . import reward as rewards

RewardConfig = rewards.RewardConfig

NO_DATA_FITNESS = 0


def evaluate_fitness(
    transitions: Iterable[dict[str, Any]],
    expected_episodes: int = 1,
    config: RewardConfig | None = None,
) -> FitnessReport:
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be at least 1")

    rules = config or rewards.RewardConfig()
    best: dict[Any, float] = {}
    total = 0.0
    kills = 0
    count = 0
    aim_sum = 0.0
    aim_ticks = 0
    episode_steps = 0
    episode_had_target = False
    completed_episodes = 0
    empty_episodes = 0

    for transition in transitions:
        count += 1
        episode_steps += 1
        if not isinstance(transition, dict):
            return FitnessReport(0.0, count, kills, error="invalid transition record")
        state = transition.get("state")
        next_state = transition.get("next_state")
        if not isinstance(state, dict) or not isinstance(next_state, dict):
            return FitnessReport(0.0, count, kills, error="missing transition state")
        for snapshot in (state, next_state):
            if rewards._offset(snapshot) is not None:
                episode_had_target = True
                if rewards._number(snapshot, "yaw") is None:
                    return FitnessReport(0.0, count, kills, error="missing target yaw")

        reward = rewards.reward_transition(transition, rules)
        total += reward.value
        kills += reward.kill_count

        if reward.aim_error is not None:
            aim_sum += reward.aim_error
            aim_ticks += 1

        if isinstance(state, dict) and isinstance(next_state, dict):
            target_id = state.get("target_id")
            if target_id is not None and target_id == next_state.get("target_id"):
                target = rewards._vector(state, "target_position")
                old_position = rewards._vector(state, "position")
                new_position = rewards._vector(next_state, "position")
                if target is not None and old_position is not None and new_position is not None:
                    d_old = math.hypot(target[0] - old_position[0], target[2] - old_position[2])
                    d_new = math.hypot(target[0] - new_position[0], target[2] - new_position[2])
                    best_dist = min(best.get(target_id, d_old), d_old)
                    gain = max(0.0, best_dist - d_new)
                    best[target_id] = min(best_dist, d_new)
                    if d_new >= rules.attack_range:
                        total += rules.progress * gain

        if transition.get("done"):
            if aim_ticks:
                total -= rules.aim * aim_sum / aim_ticks
            completed_episodes += 1
            empty_episodes += not episode_had_target
            aim_sum = 0.0
            aim_ticks = 0
            episode_steps = 0
            episode_had_target = False
            best.clear()

    if not count:
        return FitnessReport(NO_DATA_FITNESS, 0, 0, error="no transitions recorded")
    if episode_steps or completed_episodes != expected_episodes:
        return FitnessReport(
            0.0, count, kills,
            error=f"expected {expected_episodes} complete episodes, got {completed_episodes}",
        )
    if empty_episodes:
        return FitnessReport(
            0.0, count, kills, error=f"{empty_episodes} episodes contained no zombie",
        )

    return FitnessReport(total / expected_episodes, count, kills)
