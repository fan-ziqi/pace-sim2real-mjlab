"""Explicit, non-invasive environment binding for PACE's legacy optimizer API."""

from __future__ import annotations

import weakref
from typing import Any

_ENVIRONMENTS: weakref.WeakKeyDictionary[Any, weakref.ReferenceType[Any]] = (
    weakref.WeakKeyDictionary()
)


def bind_environment(env: Any) -> None:
    """Associate PACE scene entities with their owning environment.

    This makes ``CMAESOptimizer.update_simulator(robot, joint_ids, initial)``
    available without changing mjlab classes or adding private attributes to
    entities.  The registry owns no environment references, so closing and
    dropping an environment releases the association naturally.
    """
    entities = getattr(env.scene, "_entities", {})
    for entity in entities.values():
        _ENVIRONMENTS[entity] = weakref.ref(env)


def bound_environment(articulation: Any) -> Any | None:
    """Return the environment explicitly registered for an articulation."""
    reference = _ENVIRONMENTS.get(articulation)
    return None if reference is None else reference()


def install_automatic_binding() -> None:
    """Deprecated compatibility no-op.

    Older prereleases patched ``ManagerBasedRlEnv.__init__`` from this helper.
    PACE now deliberately requires explicit ``bind_environment(env)`` for a
    manually-created environment; package imports have no global side effects.
    """
