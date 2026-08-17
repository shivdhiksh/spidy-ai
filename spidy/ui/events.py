"""
UI EventBus Events — Milestone 5: Desktop Overlay UI
=====================================================
Topic namespace: ``ui.*``

These events are the ONLY way external modules communicate with the UI.
The overlay subscribes to these topics. The Brain, Voice pipeline, and
other modules publish to these topics. The UI publishes back for control flow.

Publish to UI (other modules → UI)
-----------------------------------
ui.show              → Make overlay visible (e.g. on wake word)
ui.hide              → Hide overlay
ui.state_change      → UIState transition (listening / speaking / idle)
ui.message           → Add a message bubble to chat view
ui.notify            → Show a toast notification
ui.waveform_data     → Raw amplitude array for waveform visualiser
ui.theme_change      → Switch dark ↔ light theme

Published by UI (UI → other modules)
--------------------------------------
ui.ready             → Overlay is initialised and visible
ui.closed            → User closed the overlay
ui.hotkey_pressed    → Global hotkey fired (toggle overlay)
ui.mic_button_clicked → User clicked the microphone button
ui.settings_opened   → Settings panel opened
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


# ─── Inbound (other modules → UI) ────────────────────────────────────────────


@dataclass
class UIShowEvent(Event):
    """Request the overlay to appear. Optionally position near screen edge."""
    topic = "ui.show"
    edge: str = "top-right"       # "top-right" | "top-left" | "bottom-right" | "bottom-left"
    animate: bool = True          # Slide in with animation?
    source: str = ""              # What triggered this (e.g. "wake_word", "hotkey")


@dataclass
class UIHideEvent(Event):
    """Request the overlay to hide."""
    topic = "ui.hide"
    animate: bool = True


@dataclass
class UIStateChangeEvent(Event):
    """
    Transition the overlay to a new visual state.

    State names
    -----------
    "idle"       — Resting; subtle avatar pulse only
    "wake_ready" — Listening for wake word; edge glow active
    "listening"  — Actively transcribing speech; waveform + mic pulse
    "thinking"   — LLM processing; three-dot spinner
    "speaking"   — TTS playing; audio bars animated
    "error"      — Error state; brief red flash
    """
    topic = "ui.state_change"
    state: str = "idle"
    detail: str = ""              # Optional subtitle (e.g. "Transcribing...")


@dataclass
class UIMessageEvent(Event):
    """Add a message bubble to the chat view."""
    topic = "ui.message"
    role: str = "assistant"       # "user" | "assistant" | "system"
    text: str = ""
    session_id: str = ""


@dataclass
class UINotifyEvent(Event):
    """Show a toast notification above the overlay."""
    topic = "ui.notify"
    title: str = ""
    body: str = ""
    level: str = "info"           # "info" | "success" | "warning" | "error"
    duration_ms: int = 4000       # Auto-dismiss delay; 0 = persistent


@dataclass
class UIWaveformDataEvent(Event):
    """Real-time audio amplitude data for the waveform widget."""
    topic = "ui.waveform_data"
    amplitudes: list[float] = field(default_factory=list)  # 0.0–1.0 per band


@dataclass
class UIThemeChangeEvent(Event):
    """Switch the overlay theme."""
    topic = "ui.theme_change"
    theme: str = "dark"           # "dark" | "light"


# ─── Outbound (UI → other modules) ───────────────────────────────────────────


@dataclass
class UIReadyEvent(Event):
    """Published once the overlay window is created and visible."""
    topic = "ui.ready"


@dataclass
class UIClosedEvent(Event):
    """Published when the user closes the overlay (or it exits)."""
    topic = "ui.closed"


@dataclass
class UIHotkeyPressedEvent(Event):
    """Published when the global hotkey fires (e.g. Ctrl+Space)."""
    topic = "ui.hotkey_pressed"
    hotkey: str = ""


@dataclass
class UIMicButtonClickedEvent(Event):
    """Published when the user clicks the microphone button."""
    topic = "ui.mic_button_clicked"


@dataclass
class UISettingsOpenedEvent(Event):
    """Published when the settings panel is opened."""
    topic = "ui.settings_opened"


# ─── Confirmation response events (UI → agent) ───────────────────────────────
# These are published back into the agent.* namespace so the ExecutionLoop
# can unblock without any new event infrastructure.

@dataclass
class UIConfirmationGrantedEvent(Event):
    """User confirmed a task requiring authority approval."""
    topic = "agent.confirmation_granted"
    goal_id: str = ""
    task_id: str = ""


@dataclass
class UIConfirmationDeniedEvent(Event):
    """User denied/cancelled a task requiring authority approval."""
    topic = "agent.confirmation_denied"
    goal_id: str = ""
    task_id: str = ""
