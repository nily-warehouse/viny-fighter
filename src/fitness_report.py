from dataclasses import dataclass


@dataclass(frozen=True)
class FitnessReport:
    fitness: float
    transitions: int
    inferred_kills: int
    error: str | None = None
