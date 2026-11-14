"""Spidy must never read symbols aloud, and should sound like she means it."""

from __future__ import annotations

import asyncio

import pytest

from spidy.voice.emotion import CALM, CARING, CURIOUS, EXCITED, HAPPY, SORRY, WARM, style_for
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.tts_text_filter import TTSTextFilter

F = TTSTextFilter()


@pytest.mark.parametrize("raw, spoken", [
    ("Sure! 5 + 3 = 8 and 10 * 2 = 20 */*+", "Sure! 5 plus 3 equals 8 and 10 times 2 equals 20"),
    ("CPU is at 45% and it is 24°C 🌤️", "CPU is at 45 percent and it is 24 degrees Celsius"),
    ("**Done!** Opened VS Code -> ready.", "Done! Opened VS Code to ready."),
    ("This is *really* good and _very_ fast.", "This is really good and very fast."),
    ("Steps:\n- Open *Settings*\n- Click #2\n1. Done", "Steps: Open Settings. Click number 2. Done"),
    ("It costs $5 or ₹ 400.", "It costs 5 dollars or 400 rupees."),
    ("wow!!! really???", "wow! really?"),
    ("a | b ^ c ~ {d} [e] \\ f", "a b c d e f"),
])
def test_symbols_are_spoken_as_words_or_dropped(raw, spoken):
    assert F.filter(raw).tts_text == spoken


def test_nothing_unspeakable_survives():
    out = F.filter("#$%^&*()_+{}|:<>?~`-=[]\\;',./ 🙂✅").tts_text
    assert not set(out) & set("#$^*_+{}|<>~`=[]\\/🙂✅")


@pytest.mark.parametrize("sentence, style", [
    ("Done! VS Code is open.", HAPPY),
    ("That's great news.", HAPPY),
    ("Yay, it worked!", EXCITED),
    ("Don't worry, I'm here.", CARING),
    ("Sorry, I couldn't find that file.", SORRY),
    ("Unfortunately the download failed.", SORRY),
    ("Which folder should I use?", CURIOUS),
    ("Yes boss.", WARM),
    ("The file is in your Documents folder.", CALM),
])
def test_each_sentence_gets_a_matching_style(sentence, style):
    assert style_for(sentence) == style


def test_sorry_is_slower_and_softer_and_good_news_is_livelier():
    assert SORRY.speed < CALM.speed < EXCITED.speed
    assert SORRY.volume < CALM.volume < EXCITED.volume
    assert EXCITED.expressiveness > CALM.expressiveness > SORRY.expressiveness


class _StyledEngine:
    is_speaking = False
    voice_name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def stop(self) -> None:
        pass

    async def synthesize(self, text, style=None):
        self.calls.append((text, style.name if style else ""))
        return None

    async def play_buffer(self, buf) -> None:
        pass

    async def speak(self, text) -> None:
        pass


def test_the_speaker_cleans_text_and_styles_every_sentence():
    eng = _StyledEngine()
    asyncio.run(StreamingTTSWrapper(eng).speak("**Done!** 5 + 3 = 8. Sorry, the second step failed."))
    assert eng.calls == [("Done!", "excited"), ("5 plus 3 equals 8.", "calm"),
                         ("Sorry, the second step failed.", "sorry")]
