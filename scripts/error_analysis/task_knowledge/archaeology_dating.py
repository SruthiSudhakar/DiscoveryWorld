"""Archaeology Dating: numeric ground-truth ages in critical hypotheses."""
from __future__ import annotations

import re
from typing import Any


class ArchaeologyDating:
    name = "Archaeology Dating"

    def matches(self, identifier: str) -> bool:
        ident = identifier.lower()
        return "archaeolog" in ident

    def analyze(self, rows, traj, hypotheses, subscores) -> dict[str, Any]:
        hyp = (hypotheses or [""])[0]
        hyp_ages = sorted({int(x) for x in re.findall(r"\d{4,}", hyp)})
        correct_oldest = max(hyp_ages) if hyp_ages else None

        ages_cited: set[int] = set()
        correct_oldest_cited = False
        for r in rows:
            a = r.get("action") or {}
            t = (a.get("thought") or "")
            for m in re.findall(r"\d{4,}", t):
                mi = int(m)
                if mi in hyp_ages:
                    ages_cited.add(mi)
                    if correct_oldest is not None and mi == correct_oldest:
                        correct_oldest_cited = True

        failure_modes: list[str] = []
        if correct_oldest is not None and not correct_oldest_cited:
            failure_modes.append("never_cited_correct_oldest_age")

        artifacts_dated = subscores.get("Artifacts dated", (0, 0, ""))
        flag_placed = subscores.get("Flag placed", (0, 1, ""))
        if artifacts_dated[1] and artifacts_dated[0] == artifacts_dated[1] and not correct_oldest_cited:
            failure_modes.append("dated_but_didnt_identify_oldest")
        if (
            artifacts_dated[1]
            and artifacts_dated[0] == artifacts_dated[1]
            and flag_placed[0] == 0
        ):
            failure_modes.append("wrong_oldest_pick_or_misplaced_flag")

        return {
            "failure_modes": failure_modes,
            "extra": {
                "correct_oldest_age": correct_oldest,
                "ages_cited_in_thoughts": sorted(ages_cited),
                "correct_oldest_cited": correct_oldest_cited,
            },
        }
