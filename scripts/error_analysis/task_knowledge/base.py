"""Plugin protocol + no-op default."""
from __future__ import annotations

from typing import Any, Protocol


class TaskKnowledge(Protocol):
    """A small plugin that adds task-specific failure-mode detection.

    Implementations can return any extra fields they want stored on the
    episode result via `extra`; they should also append failure-mode
    strings to `failure_modes`.
    """

    name: str

    def matches(self, identifier: str) -> bool:
        ...

    def analyze(
        self,
        rows: list[dict],
        traj: list[dict],
        hypotheses: list[str],
        subscores: dict[str, tuple[int, int, str]],
    ) -> dict[str, Any]:
        ...


class NoOpKnowledge:
    name = "generic"

    def matches(self, identifier: str) -> bool:  # pragma: no cover
        return False

    def analyze(self, rows, traj, hypotheses, subscores) -> dict[str, Any]:
        return {"failure_modes": [], "extra": {}}
