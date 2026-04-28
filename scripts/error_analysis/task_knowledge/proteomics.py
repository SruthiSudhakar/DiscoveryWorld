"""Proteomics: qualitative ground truth, but we can still flag specific
prerequisite/measurement/placement subtask failures."""
from __future__ import annotations

from typing import Any


class Proteomics:
    name = "Proteomics"

    def matches(self, identifier: str) -> bool:
        return "proteomic" in identifier.lower()

    def analyze(self, rows, traj, hypotheses, subscores) -> dict[str, Any]:
        failure_modes: list[str] = []
        used_meter = subscores.get("Use proteomics meter", (0, 0, ""))
        moved_flag = subscores.get("Move flag to correct location", (0, 1, ""))
        if used_meter[1] and used_meter[0] == used_meter[1] and moved_flag[0] == 0:
            failure_modes.append("measured_but_misplaced_flag")
        return {"failure_modes": failure_modes, "extra": {}}
