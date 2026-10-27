"""
PipelineTracer — Live Voice & Agent Pipeline Latency Instrumentation
====================================================================
Instruments nanosecond-precision timings across all 11 stages of the
SPIDY live execution pipeline:

  1. wake_ms                 — Wake word detection and debounce
  2. session_ms              — Session activation and ack delivery
  3. stt_ms                  — Whisper speech-to-text transcription
  4. transcript_cleanup_ms   — Normalization and ASR fragment recovery
  5. intent_ms               — Fast intent classification
  6. planning_ms             — Task decomposition and contract generation
  7. tool_selection_ms       — Structured router and skill discovery
  8. execution_ms            — Skill execution under DesktopStateLock
  9. observation_ms          — TaskObserver environmental verification
 10. verification_ms         — Final GoalVerifier multi-signal proof
 11. tts_ms                  — StreamingTTS response audio delivery
     total_ms                — Complete end-to-end roundtrip latency
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from spidy.core.event_bus import Event
from spidy.logging.logger import get_logger

log = get_logger(__name__)


@dataclass
class PipelineTraceEvent(Event):
    """Event published when a complete pipeline trace is finalized."""
    topic = "telemetry.pipeline_trace"
    trace_id: str = ""
    raw_transcript: str = ""
    normalized_transcript: str = ""
    decision: str = "deterministic_tool"
    tool: str = ""
    llm_required: bool = False
    llm_reason: str = ""
    llm_calls: int = 0
    wake_ms: float = 0.0
    stt_ms: float = 0.0
    transcript_cleanup_ms: float = 0.0
    intent_ms: float = 0.0
    planning_ms: float = 0.0
    tool_selection_ms: float = 0.0
    execution_ms: float = 0.0
    observation_ms: float = 0.0
    verification_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PipelineTrace:
    """Active trace container accumulating metrics across stages."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    raw_transcript: str = ""
    normalized_transcript: str = ""
    decision: str = "deterministic_tool"
    tool: str = ""
    llm_required: bool = False
    llm_reason: str = "deterministic_execution"
    llm_calls: int = 0

    wake_ms: float = 0.0
    session_ms: float = 0.0
    stt_ms: float = 0.0
    transcript_cleanup_ms: float = 0.0
    intent_ms: float = 0.0
    planning_ms: float = 0.0
    tool_selection_ms: float = 0.0
    execution_ms: float = 0.0
    observation_ms: float = 0.0
    verification_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0

    _start_time: float = field(default_factory=time.monotonic)
    _stage_starts: dict[str, float] = field(default_factory=dict)

    def start_stage(self, stage_name: str) -> None:
        """Mark start of a named pipeline stage."""
        self._stage_starts[stage_name] = time.monotonic()

    def end_stage(self, stage_name: str) -> float:
        """Mark end of a named pipeline stage and update timing."""
        start = self._stage_starts.pop(stage_name, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.monotonic() - start) * 1000.0
        attr = f"{stage_name}_ms"
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + elapsed_ms)
        return elapsed_ms

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Directly record an externally measured duration."""
        attr = f"{stage_name}_ms"
        if hasattr(self, attr):
            setattr(self, attr, duration_ms)

    def finish(self) -> PipelineTraceEvent:
        """Finalize total roundtrip duration and log trace metrics."""
        self.total_ms = (time.monotonic() - self._start_time) * 1000.0
        log.info(
            "[PIPELINE_TRACE] tx={tx} | raw='{raw}' | norm='{norm}' | decision={dec} | tool={tool} | "
            "llm_required={llm_req} (calls={calls} reason='{reason}') | "
            "wake={w:.1f}ms stt={stt:.1f}ms cleanup={cl:.1f}ms intent={intent:.1f}ms plan={plan:.1f}ms tool_select={ts:.1f}ms "
            "exec={exe:.1f}ms obs={obs:.1f}ms verify={ver:.1f}ms tts={tts:.1f}ms | total={tot:.1f}ms",
            tx=self.trace_id,
            raw=self.raw_transcript[:40],
            norm=self.normalized_transcript[:40],
            dec=self.decision,
            tool=self.tool,
            llm_req=self.llm_required,
            calls=self.llm_calls,
            reason=self.llm_reason,
            w=self.wake_ms,
            stt=self.stt_ms,
            cl=self.transcript_cleanup_ms,
            intent=self.intent_ms,
            plan=self.planning_ms,
            ts=self.tool_selection_ms,
            exe=self.execution_ms,
            obs=self.observation_ms,
            ver=self.verification_ms,
            tts=self.tts_ms,
            tot=self.total_ms,
        )
        return PipelineTraceEvent(
            trace_id=self.trace_id,
            raw_transcript=self.raw_transcript,
            normalized_transcript=self.normalized_transcript,
            decision=self.decision,
            tool=self.tool,
            llm_required=self.llm_required,
            llm_reason=self.llm_reason,
            llm_calls=self.llm_calls,
            wake_ms=self.wake_ms,
            stt_ms=self.stt_ms,
            transcript_cleanup_ms=self.transcript_cleanup_ms,
            intent_ms=self.intent_ms,
            planning_ms=self.planning_ms,
            tool_selection_ms=self.tool_selection_ms,
            execution_ms=self.execution_ms,
            observation_ms=self.observation_ms,
            verification_ms=self.verification_ms,
            tts_ms=self.tts_ms,
            total_ms=self.total_ms,
        )


class PipelineTracer:
    """Global manager for creating and tracking active pipeline traces."""

    _active_trace: PipelineTrace | None = None

    @classmethod
    def start_trace(cls, raw_transcript: str = "") -> PipelineTrace:
        trace = PipelineTrace(raw_transcript=raw_transcript)
        cls._active_trace = trace
        return trace

    @classmethod
    def get_current(cls) -> PipelineTrace | None:
        return cls._active_trace

    @classmethod
    def clear(cls) -> None:
        cls._active_trace = None
