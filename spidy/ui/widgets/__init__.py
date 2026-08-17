"""spidy.ui.widgets package -- All overlay UI widgets."""

# Legacy M17.1 widgets (still used by existing tests)
from spidy.ui.widgets.chat_view        import ChatMessage, ChatView
from spidy.ui.widgets.confirmation_card import ConfirmationCard
from spidy.ui.widgets.mic_button       import MicrophoneButton
from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
from spidy.ui.widgets.spidy_orb        import SpidyOrb
from spidy.ui.widgets.task_panel       import TaskPanel, TaskStep as _TaskStepLegacy
from spidy.ui.widgets.waveform         import WaveformWidget

# M17.2 HUD widgets
from spidy.ui.widgets.hud_core         import SpidyCoreWidget
from spidy.ui.widgets.hud_telemetry    import TelemetryPanel
from spidy.ui.widgets.hud_chat         import HUDChatOverlay
from spidy.ui.widgets.hud_voice_bar    import HUDVoiceBar
from spidy.ui.widgets.hud_task_console import HUDTaskConsole, TaskStep
from spidy.ui.widgets.hud_confirmation import HUDConfirmation

__all__ = [
    # Legacy
    "ChatMessage", "ChatView",
    "ConfirmationCard",
    "MicrophoneButton",
    "SpeakingIndicator",
    "SpidyOrb",
    "TaskPanel",
    "WaveformWidget",
    # HUD
    "SpidyCoreWidget",
    "TelemetryPanel",
    "HUDChatOverlay",
    "HUDVoiceBar",
    "HUDTaskConsole",
    "TaskStep",
    "HUDConfirmation",
]
