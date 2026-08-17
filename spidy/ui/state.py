"""
UI State Machine — Milestone 5 / Milestone 17
==============================================
Defines the visual states Spidy's overlay can be in and the allowed
transitions between them.

State diagram
-------------

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   IDLE ←──────────────────────────────────── SPEAKING      │
    │     │     ↑                                      ↑         │
    │     ↓     │                                      │         │
    │  WAKE_READY ──→ LISTENING ──→ THINKING ───────────┘        │
    │     ↑               │              │                       │
    │     └───────────────┘              ↓                       │
    │                               WORKING ──→ IDLE             │
    │                                                             │
    │   ERROR (reachable from any state, returns to IDLE)         │
    └─────────────────────────────────────────────────────────────┘

State meanings
--------------
IDLE        Overlay is resting. Subtle avatar pulse. Low visual weight.
WAKE_READY  Ready for wake-word detection. Edge glow active.
            (Prepared for M6 wake-word integration — not yet connected.)
LISTENING   Actively capturing speech. Waveform animated. Mic button red.
THINKING    LLM is processing. Three-dot spinner animation.
SPEAKING    TTS is playing. Audio bars animated. Mic button disabled.
WORKING     Autonomous agent executing a goal. Task progress visible.
ERROR       Brief error flash. Automatically reverts to IDLE after timeout.
"""

from __future__ import annotations

from enum import Enum, auto


class UIState(str, Enum):
    """Visual state of the Spidy overlay."""

    IDLE = "idle"
    WAKE_READY = "wake_ready"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WORKING = "working"   # Autonomous agent is executing a goal
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


# ── Allowed transitions ───────────────────────────────────────────────────────

# Maps each state to the set of states it may transition to.
# Error can always be reached from any state.
_ALLOWED: dict[UIState, set[UIState]] = {
    UIState.IDLE:       {UIState.WAKE_READY, UIState.LISTENING, UIState.WORKING, UIState.ERROR},
    UIState.WAKE_READY: {UIState.IDLE, UIState.LISTENING, UIState.WORKING, UIState.ERROR},
    UIState.LISTENING:  {UIState.IDLE, UIState.WAKE_READY, UIState.THINKING, UIState.WORKING, UIState.ERROR},
    UIState.THINKING:   {UIState.IDLE, UIState.SPEAKING, UIState.WORKING, UIState.ERROR},
    UIState.SPEAKING:   {UIState.IDLE, UIState.LISTENING, UIState.WORKING, UIState.ERROR},
    UIState.WORKING:    {UIState.IDLE, UIState.LISTENING, UIState.ERROR},
    UIState.ERROR:      {UIState.IDLE},
}

# Allow ERROR transition from any state
for _s in UIState:
    _ALLOWED.setdefault(_s, set()).add(UIState.ERROR)


class UIStateTransitionError(ValueError):
    """Raised when an illegal state transition is attempted."""


class UIStateMachine:
    """
    Thread-safe (read-only from any thread) UI state machine.

    The machine enforces allowed transitions and tracks the
    previous state for animation direction.

    Parameters
    ----------
    initial:
        Starting state. Defaults to IDLE.
    strict:
        If True, raises UIStateTransitionError on illegal transitions.
        If False, logs a warning and ignores the transition.
    """

    def __init__(
        self,
        initial: UIState = UIState.IDLE,
        strict: bool = False,
    ) -> None:
        self._state = initial
        self._previous = initial
        self._strict = strict
        self._listeners: list = []  # list[Callable[[UIState, UIState], None]]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def current(self) -> UIState:
        return self._state

    @property
    def previous(self) -> UIState:
        return self._previous

    @property
    def is_idle(self) -> bool:
        return self._state == UIState.IDLE

    @property
    def is_listening(self) -> bool:
        return self._state == UIState.LISTENING

    @property
    def is_speaking(self) -> bool:
        return self._state == UIState.SPEAKING

    @property
    def is_thinking(self) -> bool:
        return self._state == UIState.THINKING

    @property
    def is_working(self) -> bool:
        return self._state == UIState.WORKING

    @property
    def is_active(self) -> bool:
        """True when Spidy is doing something (not idle or wake_ready)."""
        return self._state not in (UIState.IDLE, UIState.WAKE_READY)

    # ── Transitions ───────────────────────────────────────────────────────

    def transition(self, new_state: UIState | str) -> bool:
        """
        Attempt a state transition.

        Parameters
        ----------
        new_state:
            Target state (UIState enum or string name).

        Returns
        -------
        bool
            True if the transition was performed; False if rejected.

        Raises
        ------
        UIStateTransitionError
            If strict mode is enabled and the transition is illegal.
        """
        if isinstance(new_state, str):
            try:
                new_state = UIState(new_state)
            except ValueError:
                raise UIStateTransitionError(
                    f"Unknown UI state: '{new_state}'. "
                    f"Valid states: {[s.value for s in UIState]}"
                )

        if new_state == self._state:
            return True  # No-op, not an error

        allowed = _ALLOWED.get(self._state, set())
        if new_state not in allowed:
            msg = (
                f"Illegal UI state transition: {self._state.value!r} → {new_state.value!r}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
            if self._strict:
                raise UIStateTransitionError(msg)
            # In lenient mode, only reject truly nonsensical transitions
            # and log. We allow most transitions for resilience.
            from spidy.logging.logger import get_logger
            get_logger(__name__).warning(msg)

        old = self._state
        self._previous = old
        self._state = new_state
        self._notify(old, new_state)
        return True

    def can_transition_to(self, target: UIState) -> bool:
        """Return True if transitioning to target is currently allowed."""
        return target in _ALLOWED.get(self._state, set()) or target == UIState.ERROR

    def add_listener(self, fn) -> None:
        """
        Register a callback invoked on every state change.

        Signature: ``fn(old_state: UIState, new_state: UIState) -> None``
        """
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        self._listeners = [l for l in self._listeners if l is not fn]

    def reset(self) -> None:
        """Reset to IDLE without notifying listeners."""
        self._previous = self._state
        self._state = UIState.IDLE

    # ── Internal ──────────────────────────────────────────────────────────

    def _notify(self, old: UIState, new: UIState) -> None:
        for fn in list(self._listeners):
            try:
                fn(old, new)
            except Exception:  # noqa: BLE001
                pass

    def __repr__(self) -> str:
        return (
            f"UIStateMachine(current={self._state.value!r}, "
            f"previous={self._previous.value!r})"
        )
