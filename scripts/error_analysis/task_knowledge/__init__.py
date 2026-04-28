"""Per-task knowledge plugins for error analysis.

Generic analysis lives in `analyze_run.py`. Anything that requires
task-specific assumptions (e.g. the answer is a number, or a particular
ordering of subtasks matters) goes here.

Dispatch is by `scoreCard[0]["taskName"]` (e.g. "ArchaeologyDigTaskEasy") OR
the run-dir basename ("Archaeology_Dating", "Proteomics"). The first match
wins; otherwise we return a no-op plugin.
"""
from __future__ import annotations

from .archaeology_dating import ArchaeologyDating
from .base import NoOpKnowledge, TaskKnowledge
from .proteomics import Proteomics

_PLUGINS: list[TaskKnowledge] = [ArchaeologyDating(), Proteomics()]


def for_task(task_name: str | None, run_dir_basename: str | None = None) -> TaskKnowledge:
    candidates = [s for s in (task_name, run_dir_basename) if s]
    for plugin in _PLUGINS:
        for cand in candidates:
            if plugin.matches(cand):
                return plugin
    return NoOpKnowledge()


__all__ = ["for_task", "TaskKnowledge", "NoOpKnowledge"]
