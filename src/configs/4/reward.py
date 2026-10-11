from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...actions import attack_index


ATTACK_ACTION = attack_index(Path(__file__).resolve().with_name("actions.json"))


@dataclass(frozen=True)
class RewardConfig:
    attack_action: int | None = ATTACK_ACTION
    progress: float = 0
    attack_range: float = 2.5
    kill: float = 1
    aimed_attack: float = 0  # fallback
    damage_dealt: float = 1
    damage: float = 0
    aim: float = 0.05  # max episode-level penalty for yaw error (applied in fitness.py)


@dataclass(frozen=True)
class Reward:
    value: float
    inferred_kill: bool
    aim_error: float | None = None  # normalized yaw error in [0, 1]; None if no target/yaw
    kill_count: int = 0


def _vector(state: Any, key: str) -> tuple[float, float, float] | None:
    if not isinstance(state, dict):
        return None
    try:
        x, y, z = (float(value) for value in state.get(key))
    except (TypeError, ValueError):
        return None
    return (x, y, z) if all(math.isfinite(value) for value in (x, y, z)) else None


def _offset(state: Any) -> tuple[float, float, float] | None:
    return _vector(state, "target")


def _number(state: Any, key: str) -> float | None:
    if not isinstance(state, dict):
        return None
    try:
        value = float(state.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _health(state: Any) -> float | None:
    return _number(state, "health")


def _target_health(state: Any) -> float | None:
    return _number(state, "target_health")


def _same_target(state: Any, next_state: Any) -> bool:
    if not isinstance(state, dict) or not isinstance(next_state, dict):
        return False
    old_id = state.get("target_id")
    return old_id is not None and old_id == next_state.get("target_id")


def _yaw_error(state: Any) -> float | None:
    """|needed yaw - current yaw|, wrapped to [-pi, pi] and normalized to [0, 1]."""
    offset = _offset(state)
    yaw = _number(state, "yaw")  # yaw key in your state (radians)
    if offset is None or yaw is None:
        return None
    dx, _, dz = offset
    if dx == 0 and dz == 0:
        return None
    # Mineflayer convention: yaw = atan2(-dx, -dz)
    needed = math.atan2(-dx, -dz)
    diff = (needed - yaw + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    return abs(diff) / math.pi


def reward_transition(transition: dict[str, Any], config: RewardConfig | None = None) -> Reward:
    rules = config or RewardConfig()
    state = transition.get("state")
    next_state = transition.get("next_state")

    old_offset = _offset(state)

    same_target = _same_target(state, next_state)
    old_th = _target_health(state)
    new_th = _target_health(next_state)

    value = 0.0
    kill_count = 0
    events = next_state.get("combat_events") if isinstance(next_state, dict) else None
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            damage = _number(event, "damage")
            if damage is not None:
                value += rules.damage_dealt * max(0.0, damage)
            if event.get("killed") is True:
                kill_count += 1
                value += rules.kill
    else:
        killed = (
            old_offset is not None and isinstance(next_state, dict) and (
                next_state.get("target_killed") is True
                or (same_target and old_th is not None and old_th > 0
                    and new_th is not None and new_th <= 0)
            )
        )
        if killed:
            kill_count = 1
            value += rules.kill

        if same_target and old_th is not None and new_th is not None:
            value += rules.damage_dealt * max(0.0, old_th - new_th)
        elif killed and old_th is not None:
            value += rules.damage_dealt * max(0.0, old_th)
        elif (
            rules.attack_action is not None
            and transition.get("action") == rules.attack_action
            and isinstance(next_state, dict)
            and next_state.get("attack_hit") is True
        ):
            value += rules.aimed_attack

    # Damage taken
    old_health = _health(state)
    new_health = _health(next_state)
    if old_health is not None and new_health is not None:
        value += rules.damage * max(0.0, old_health - new_health)

    return Reward(
        value=value, inferred_kill=kill_count > 0,
        aim_error=_yaw_error(next_state), kill_count=kill_count,
    )
