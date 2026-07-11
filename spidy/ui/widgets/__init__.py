"""spidy.ui.widgets package — All overlay UI widgets."""

from spidy.ui.widgets.chat_view import ChatMessage, ChatView
from spidy.ui.widgets.mic_button import MicrophoneButton
from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
from spidy.ui.widgets.waveform import WaveformWidget

__all__ = [
    "ChatMessage",
    "ChatView",
    "MicrophoneButton",
    "SpeakingIndicator",
    "WaveformWidget",
]
