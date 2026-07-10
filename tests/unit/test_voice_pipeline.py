"""
Unit tests for voice pipeline ABCs and data structures.
========================================================
Tests TranscriptResult, AudioBuffer, SkillCapability, SkillResult,
and the WakeWordModel/SpeechRecognizer/TTSEngine contracts via
lightweight fake implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from spidy.perception.voice.stt.base import TranscriptResult, SpeechRecognizer
from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine
from spidy.perception.voice.wake_word.base import WakeWordModel
from spidy.skills.base import (
    BaseSkill, ParamSchema, SkillCapability, SkillContext, SkillResult,
)


# ─── Fake implementations for ABC testing ─────────────────────────────────────

class FakeWakeWordModel(WakeWordModel):
    """Minimal WakeWordModel for testing the ABC contract."""
    def __init__(self, score: float = 0.0) -> None:
        self._score = score
        self._loaded = False

    def load(self, model_path=None) -> None:
        self._loaded = True

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        return self._score

    def unload(self) -> None:
        self._loaded = False

    @property
    def model_name(self) -> str:
        return "fake_wake_word"

    @property
    def chunk_size(self) -> int:
        return 1280


class FakeSpeechRecognizer(SpeechRecognizer):
    """Minimal SpeechRecognizer for testing the ABC contract."""
    def __init__(self, transcript: str = "hello") -> None:
        self._transcript = transcript
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    async def transcribe(self, audio_data: np.ndarray) -> TranscriptResult:
        return TranscriptResult(text=self._transcript)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "fake_stt"


class FakeTTSEngine(TTSEngine):
    """Minimal TTSEngine for testing the ABC contract."""
    def __init__(self) -> None:
        self._speaking = False
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    async def synthesize(self, text: str) -> AudioBuffer:
        samples = np.zeros(22050, dtype=np.float32)  # 1 second of silence
        return AudioBuffer(samples=samples, sample_rate=22050, duration_seconds=1.0)

    async def speak(self, text: str) -> None:
        self._speaking = True
        self._speaking = False

    def stop(self) -> None:
        self._speaking = False

    @property
    def voice_name(self) -> str:
        return "fake_voice"

    @property
    def is_speaking(self) -> bool:
        return self._speaking


class FakeSkill(BaseSkill):
    """Minimal BaseSkill for testing the ABC contract."""
    name = "fake_skill"
    version = "1.0.0"

    def capabilities(self):
        return [
            SkillCapability(
                action="do_thing",
                description="Does a thing",
                permission_tier="T0",
                params=[ParamSchema("target", "string", required=True)],
            )
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "do_thing":
            return SkillResult.ok(f"Did {context.get('target')}")
        return SkillResult.fail(f"Unknown: {action}")


# ─── TranscriptResult tests ───────────────────────────────────────────────────

class TestTranscriptResult:
    def test_strips_whitespace(self):
        r = TranscriptResult(text="  hello world  ")
        assert r.text == "hello world"

    def test_empty_text_sets_is_empty(self):
        r = TranscriptResult(text="")
        assert r.is_empty is True

    def test_whitespace_only_sets_is_empty(self):
        r = TranscriptResult(text="   ")
        assert r.is_empty is True

    def test_non_empty_text_not_empty(self):
        r = TranscriptResult(text="open chrome")
        assert r.is_empty is False
        assert r.text == "open chrome"

    def test_default_confidence_is_one(self):
        r = TranscriptResult(text="test")
        assert r.confidence == 1.0


# ─── AudioBuffer tests ────────────────────────────────────────────────────────

class TestAudioBuffer:
    def test_stores_samples_and_rate(self):
        samples = np.ones(22050, dtype=np.float32)
        buf = AudioBuffer(samples=samples, sample_rate=22050, duration_seconds=1.0)
        assert buf.sample_rate == 22050
        assert buf.duration_seconds == 1.0
        assert len(buf.samples) == 22050


# ─── WakeWordModel ABC tests ──────────────────────────────────────────────────

class TestWakeWordModelABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            WakeWordModel()  # type: ignore

    def test_fake_implementation_works(self):
        model = FakeWakeWordModel(score=0.9)
        model.load()
        assert model.process_chunk(np.zeros(1280)) == 0.9
        assert model.chunk_size == 1280
        assert model.model_name == "fake_wake_word"
        model.unload()


# ─── SpeechRecognizer ABC tests ───────────────────────────────────────────────

class TestSpeechRecognizerABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            SpeechRecognizer()  # type: ignore

    @pytest.mark.asyncio
    async def test_fake_returns_transcript(self):
        rec = FakeSpeechRecognizer(transcript="open VS Code")
        rec.load()
        result = await rec.transcribe(np.zeros(16000))
        assert result.text == "open VS Code"
        assert not result.is_empty


# ─── TTSEngine ABC tests ──────────────────────────────────────────────────────

class TestTTSEngineABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            TTSEngine()  # type: ignore

    @pytest.mark.asyncio
    async def test_fake_synthesize_returns_buffer(self):
        tts = FakeTTSEngine()
        tts.load()
        buf = await tts.synthesize("Hello Spidy")
        assert isinstance(buf, AudioBuffer)
        assert buf.sample_rate == 22050
        assert len(buf.samples) > 0

    @pytest.mark.asyncio
    async def test_fake_speak_and_stop(self):
        tts = FakeTTSEngine()
        tts.load()
        await tts.speak("Hello")
        assert not tts.is_speaking
        tts.stop()  # must not raise


# ─── BaseSkill tests ──────────────────────────────────────────────────────────

class TestBaseSkill:
    def test_capabilities_returns_list(self):
        skill = FakeSkill()
        caps = skill.capabilities()
        assert len(caps) == 1
        assert caps[0].action == "do_thing"

    def test_supports_known_action(self):
        skill = FakeSkill()
        assert skill.supports("do_thing") is True

    def test_does_not_support_unknown_action(self):
        skill = FakeSkill()
        assert skill.supports("fly_to_moon") is False

    @pytest.mark.asyncio
    async def test_execute_known_action(self):
        skill = FakeSkill()
        ctx = SkillContext(action="do_thing", params={"target": "Downloads"})
        result = await skill.execute("do_thing", ctx)
        assert result.success is True
        assert "Downloads" in result.message

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        skill = FakeSkill()
        ctx = SkillContext(action="unknown", params={})
        result = await skill.execute("unknown", ctx)
        assert result.success is False


class TestSkillResult:
    def test_ok_factory(self):
        r = SkillResult.ok("Done!", data=[1, 2, 3])
        assert r.success is True
        assert r.message == "Done!"
        assert r.data == [1, 2, 3]
        assert r.error is None

    def test_fail_factory(self):
        exc = ValueError("oops")
        r = SkillResult.fail("Failed!", error=exc)
        assert r.success is False
        assert r.error is exc


class TestSkillContext:
    def test_get_param_with_default(self):
        ctx = SkillContext(action="test", params={"key": "val"})
        assert ctx.get("key") == "val"
        assert ctx.get("missing", "default") == "default"
        assert ctx.get("missing") is None
