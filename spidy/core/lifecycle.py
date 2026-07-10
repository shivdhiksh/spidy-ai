"""
Spidy Application Lifecycle
============================
Defines the lifecycle states and transition model for SpidyCore.

States
------
CREATED → INITIALISING → READY → RUNNING → STOPPING → STOPPED → ERROR

Every module and the UI can subscribe to lifecycle events to perform
setup and teardown cleanly.
"""

from __future__ import annotations

from enum import Enum, auto


class LifecycleState(Enum):
    """
    Canonical application lifecycle states.

    Transitions
    -----------
    CREATED      : Object constructed, nothing started.
    INITIALISING : Modules loading, config parsed, loggers configured.
    READY        : All modules loaded. Voice pipeline not yet active.
    RUNNING      : Fully operational. Wake word detector is active.
    STOPPING     : Graceful shutdown in progress.
    STOPPED      : All resources released, safe to exit.
    ERROR        : Unrecoverable failure. Requires restart.
    """

    CREATED = auto()
    INITIALISING = auto()
    READY = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

    def can_transition_to(self, new_state: "LifecycleState") -> bool:
        """Return True if a transition from self → new_state is valid."""
        valid_transitions: dict[LifecycleState, set[LifecycleState]] = {
            LifecycleState.CREATED:       {LifecycleState.INITIALISING, LifecycleState.ERROR},
            LifecycleState.INITIALISING:  {LifecycleState.READY, LifecycleState.ERROR},
            LifecycleState.READY:         {LifecycleState.RUNNING, LifecycleState.STOPPING, LifecycleState.ERROR},
            LifecycleState.RUNNING:       {LifecycleState.STOPPING, LifecycleState.ERROR},
            LifecycleState.STOPPING:      {LifecycleState.STOPPED, LifecycleState.ERROR},
            LifecycleState.STOPPED:       set(),
            LifecycleState.ERROR:         {LifecycleState.STOPPING},
        }
        return new_state in valid_transitions.get(self, set())
