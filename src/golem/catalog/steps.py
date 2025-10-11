"""Utilities for assembling pipeline steps from declarative specs."""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import golem as sps


def build_steps(step_specs: List[Dict[str, Any]]) -> List[Tuple[callable, Dict[str, Any]]]:
    """Convert a list of step specifications into callables for run_project_pipeline.

    Each ``step_specs`` entry must contain a ``name`` (function name) and the
    keyword arguments to pass. ``dataset:`` and ``dataset_meta:`` markers are
    resolved later by :func:`golem.catalog.run_project_pipeline`.
    """
    steps: List[Tuple[callable, Dict[str, Any]]] = []
    for spec in step_specs:
        try:
            func = getattr(sps, spec["name"])
        except AttributeError as exc:
            raise ValueError(f"Unknown pipeline step: {spec['name']}") from exc
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        steps.append((func, kwargs))
    return steps


__all__ = ["build_steps"]
