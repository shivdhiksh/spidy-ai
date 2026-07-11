"""
SpidyCore — Application Bootstrap & Orchestrator
=================================================
The root object that owns all modules and drives the application lifecycle.

Responsibilities
----------------
1. Parse & validate configuration (ConfigManager)
2. Configure the logging system
3. Instantiate the event bus
4. Instantiate and initialise each module in dependency order
5. Start the main run loop
6. Handle graceful shutdown (SIGINT, SIGTERM, Windows CTRL+C)

Architecture
------------
SpidyCore is NOT a singleton. In tests, you can create multiple instances
with different configs. In production, main.py creates exactly one and runs it.

Module Dependency Order (bottom-up)
------------------------------------
1. Logger          (no deps)
2. ConfigManager   (no deps)
3. EventBus        (no deps)
4. DeviceManager   (CUDA detection)
5. VoiceEngine     (wake word, STT, TTS)
6. [Future] Brain, MemoryEngine, UI, Skills...


Example
-------
    import asyncio
    from spidy.core.app import SpidyCore

    async def main():
        core = SpidyCore()
        await core.start()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from spidy.config.manager import ConfigManager, SpidyConfig
from spidy.core.device import DeviceManager
from spidy.core.event_bus import EventBus, SpidyStartedEvent, SpidyShuttingDownEvent
from spidy.core.lifecycle import LifecycleState
from spidy.logging import configure as configure_logging
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.brain import Brain
    from spidy.perception.voice.engine import VoiceEngine
    from spidy.perception.context.observer_manager import ObserverManager

log = get_logger(__name__)


class SpidyCore:
    """
    Root application object. Owns all module instances.

    Parameters
    ----------
    config_path:
        Path to the YAML config file. Defaults to ``config/spidy_config.yaml``
        relative to the current working directory.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._state = LifecycleState.CREATED
        self._config_path = config_path
        self._config_mgr: ConfigManager | None = None
        self._settings: SpidyConfig | None = None
        self._bus: EventBus | None = None
        self._device_mgr: DeviceManager | None = None
        self._voice_engine: VoiceEngine | None = None
        self._observer_mgr: ObserverManager | None = None
        self._brain: Brain | None = None
        self._shutdown_event = asyncio.Event()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def settings(self) -> SpidyConfig:
        if self._settings is None:
            raise RuntimeError("SpidyCore not initialised. Call start() first.")
        return self._settings

    @property
    def bus(self) -> EventBus:
        if self._bus is None:
            raise RuntimeError("SpidyCore not initialised. Call start() first.")
        return self._bus

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Full application start sequence.

        Drives the state machine from CREATED → RUNNING.
        Blocks until shutdown is requested.
        """
        try:
            await self._initialise()
            await self._run()
        except Exception as exc:
            self._transition(LifecycleState.ERROR)
            log.critical("Fatal error during startup: {exc}", exc=exc)
            raise
        finally:
            await self._stop()

    async def request_shutdown(self) -> None:
        """
        Request graceful shutdown from anywhere in the application.

        This is the only correct way to stop Spidy. It publishes the
        shutdown event so all modules can clean up.
        """
        log.info("Shutdown requested.")
        if self._bus:
            await self._bus.publish(SpidyShuttingDownEvent())
        self._shutdown_event.set()

    # ── Private implementation ────────────────────────────────────────────

    def _transition(self, new_state: LifecycleState) -> None:
        """Validate and apply a state transition."""
        if not self._state.can_transition_to(new_state):
            raise RuntimeError(
                f"Invalid lifecycle transition: {self._state} → {new_state}"
            )
        log.debug(
            "Lifecycle: {old} → {new}",
            old=self._state.name,
            new=new_state.name,
        )
        self._state = new_state

    async def _initialise(self) -> None:
        """Step 1: Bootstrap all foundation services."""
        self._transition(LifecycleState.INITIALISING)
        log.info("=" * 60)
        log.info("  Spidy is starting up...")
        log.info("=" * 60)

        # ── 1. Configuration ──────────────────────────────────────────────
        self._config_mgr = ConfigManager(config_path=self._config_path)
        self._settings = self._config_mgr.load()

        # ── 2. Logging (reconfigure with values from config) ──────────────
        log_dir = self._settings.paths.resolve("log_dir")
        configure_logging(
            level=self._settings.logging.level,
            log_dir=log_dir,
            rotation=self._settings.logging.rotation,
            retention=self._settings.logging.retention,
            colorize=self._settings.logging.colorize,
            backtrace=self._settings.logging.backtrace,
            diagnose=self._settings.logging.diagnose and self._settings.app.debug,
        )

        # ── 3. Event Bus ──────────────────────────────────────────────────
        self._bus = EventBus()
        loop = asyncio.get_running_loop()
        self._bus.set_loop(loop)

        # ── 4. Device Manager (CUDA detection) ───────────────────────────
        self._device_mgr = DeviceManager()
        self._device_mgr.detect()

        # ── 5. OS Signal Handlers (CTRL+C, SIGTERM) ───────────────────────
        self._register_signal_handlers()

        self._transition(LifecycleState.READY)
        log.info(
            "Spidy v{v} ready | Hello, {name}!",
            v=self._settings.app.version,
            name=self._settings.app.user_name,
        )

    async def _run(self) -> None:
        """Step 2: Start all services and enter the main event loop."""
        self._transition(LifecycleState.RUNNING)

        # ── Start VoiceEngine (if configured) ────────────────────────────
        if self._settings and self._settings.voice.wake_word.enabled:
            try:
                from spidy.perception.voice.factory import VoiceEngineFactory
                self._voice_engine = VoiceEngineFactory.build(
                    self._settings, self._bus
                )
                await self._voice_engine.start()
                log.info("VoiceEngine running. Say the wake word to activate Spidy.")
            except Exception as exc:
                log.warning(
                    "VoiceEngine failed to start (non-fatal): {exc}. "
                    "Running without voice.",
                    exc=exc,
                )
                self._voice_engine = None

        # ── Start ObserverManager (if context monitoring enabled) ─────────
        if self._settings and self._settings.context.enabled:
            try:
                from spidy.perception.context.observer_manager import ObserverManager
                self._observer_mgr = ObserverManager(
                    bus=self._bus,
                    config=self._settings.context,
                )
                await self._observer_mgr.start()
                log.info("ObserverManager running. Desktop context is active.")
            except Exception as exc:
                log.warning(
                    "ObserverManager failed to start (non-fatal): {exc}. "
                    "Running without context observation.",
                    exc=exc,
                )
                self._observer_mgr = None

        # ── Start Brain (always on, if reasoning config present) ──────────
        if self._settings:
            try:
                from spidy.brain.brain import Brain
                from spidy.llm.client import LLMClientFactory
                from spidy.skills.registry import SkillRegistry

                skill_registry = SkillRegistry()
                llm_client = LLMClientFactory.build(self._settings.reasoning)

                self._brain = Brain(
                    bus=self._bus,
                    config=self._settings.reasoning,
                    skill_registry=skill_registry,
                    llm_client=llm_client,
                    user_name=self._settings.app.user_name,
                )
                await self._brain.start()
                log.info("Brain running. Ready to process utterances.")
            except Exception as exc:
                log.warning(
                    "Brain failed to start (non-fatal): {exc}. Running without Brain.",
                    exc=exc,
                )
                self._brain = None

        # Announce to all modules that we are live
        await self._bus.publish(SpidyStartedEvent())

        log.info("Spidy is running. Waiting for shutdown signal...")
        # Block until shutdown is requested
        await self._shutdown_event.wait()

    async def _stop(self) -> None:
        """Step 3: Graceful shutdown sequence."""
        if self._state in (LifecycleState.STOPPED, LifecycleState.STOPPING):
            return  # Already shutting down

        self._transition(LifecycleState.STOPPING)
        log.info("Stopping Spidy...")

        # Stop Brain
        if self._brain is not None:
            await self._brain.stop()

        # Stop ObserverManager
        if self._observer_mgr is not None:
            await self._observer_mgr.stop()

        # Stop VoiceEngine first (releases audio resources)
        if self._voice_engine is not None:
            await self._voice_engine.stop()

        if self._bus:
            log.debug("Event bus stats: {stats}", stats=self._bus.stats)

        self._transition(LifecycleState.STOPPED)
        log.info("Spidy stopped cleanly. Goodbye, {name}.",
                 name=self._settings.app.user_name if self._settings else "")


    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown on Windows."""
        loop = asyncio.get_running_loop()

        def _handle_signal() -> None:
            log.warning("Interrupt signal received.")
            asyncio.ensure_future(self.request_shutdown())

        # Windows supports SIGINT (CTRL+C) and SIGBREAK (CTRL+Break)
        # SIGTERM is not natively supported on Windows but we try anyway.
        for sig in (signal.SIGINT,):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, OSError):
                # Windows asyncio event loop may not support add_signal_handler
                # Fall back to the built-in KeyboardInterrupt handling
                log.debug(
                    "Signal handler for {sig} not supported on this platform. "
                    "Using KeyboardInterrupt fallback.",
                    sig=sig,
                )
