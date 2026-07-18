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
5. Start the main run loop (voice-driven or text-input REPL)
6. Handle graceful shutdown (SIGINT, SIGTERM, Windows CTRL+C)

Architecture
------------
SpidyCore is NOT a singleton. In tests, you can create multiple instances
with different configs. In production, main.py creates exactly one and runs it.

Module Dependency Order (bottom-up)
------------------------------------
 1. Logger           (no deps)
 2. ConfigManager    (no deps)
 3. EventBus         (no deps)
 4. DeviceManager    (CUDA detection)
 5. MemoryManager    (EventBus, config)       Milestone 8
 6. VisionManager    (EventBus, config)       Milestone 9
 7. KnowledgeManager (EventBus, config)       Milestone 10
 8. VoiceEngine      (wake word, STT, TTS)    Milestone 1
 9. ObserverManager  (desktop context)        Milestone 2
10. Brain            (all of the above)       Milestone 3
11. SpidyApp / Qt    (EventBus, asyncio loop) Milestone 5

Run modes
---------
production (default):
    Brain listens for ``voice.user_spoke`` events from VoiceEngine.
    Qt overlay is shown if ``ui.enabled`` is True.

text mode (--text-mode):
    An asyncio stdin REPL loop sends input directly to ``Brain.process()``.
    Voice pipeline is not started. Useful for testing without audio hardware.

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
import threading
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
    from spidy.memory.manager import MemoryManager
    from spidy.vision.manager import VisionManager
    from spidy.knowledge.manager import KnowledgeManager
    from spidy.learning.manager import LearningManager
    from spidy.perception.voice.engine import VoiceEngine
    from spidy.perception.context.observer_manager import ObserverManager
    from spidy.plugins.manager import PluginManager
    from spidy.ui.app import SpidyApp
    from spidy.ui.brain_bridge import BrainUIBridge

log = get_logger(__name__)


class SpidyCore:
    """
    Root application object. Owns all module instances.

    Parameters
    ----------
    config_path:
        Path to the YAML config file. Defaults to ``config/spidy_config.yaml``
        relative to the current working directory.
    text_mode:
        When True, skip the voice pipeline and open an interactive text REPL
        that feeds input directly into the Brain. Useful for testing without
        audio hardware.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        text_mode: bool = False,
    ) -> None:
        self._state = LifecycleState.CREATED
        self._config_path = config_path
        self._text_mode = text_mode
        self._config_mgr: ConfigManager | None = None
        self._settings: SpidyConfig | None = None
        self._bus: EventBus | None = None
        self._device_mgr: DeviceManager | None = None
        self._voice_engine: VoiceEngine | None = None
        self._observer_mgr: ObserverManager | None = None
        self._brain: Brain | None = None
        self._memory_mgr: MemoryManager | None = None
        self._vision_mgr: VisionManager | None = None
        self._knowledge_mgr: KnowledgeManager | None = None
        self._learning_mgr: LearningManager | None = None
        self._plugin_mgr: "PluginManager | None" = None
        self._ui_app: SpidyApp | None = None
        self._ui_thread: threading.Thread | None = None
        self._brain_ui_bridge: "BrainUIBridge | None" = None
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

    @property
    def brain(self) -> "Brain | None":
        """The active Brain instance (None until _run() completes startup)."""
        return self._brain

    @property
    def memory(self) -> "MemoryManager | None":
        """The active MemoryManager (None if disabled or failed to start)."""
        return self._memory_mgr

    @property
    def vision(self) -> "VisionManager | None":
        """The active VisionManager (None if disabled or failed to start)."""
        return self._vision_mgr

    @property
    def knowledge(self) -> "KnowledgeManager | None":
        """The active KnowledgeManager (None if disabled or failed to start)."""
        return self._knowledge_mgr

    @property
    def learning(self) -> "LearningManager | None":
        """The active LearningManager (None if disabled or failed to start)."""
        return self._learning_mgr

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
        if self._text_mode:
            log.info("  Mode: TEXT REPL (voice pipeline disabled)")
        log.info("=" * 60)

        # ── 1. Configuration ──────────────────────────────────────────────
        self._config_mgr = ConfigManager(config_path=self._config_path)
        self._settings = self._config_mgr.load()

        # ── 1a. Ollama health check (onboarding diagnostics) ──────────────
        await self._run_ollama_health_check()

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

        # ── 6. Memory Engine (Milestone 8) ────────────────────────────────
        if self._settings.memory.enabled:
            try:
                from spidy.memory.manager import MemoryManager
                memory_dir = self._settings.paths.resolve("memory_dir")
                memory_dir.mkdir(parents=True, exist_ok=True)
                self._memory_mgr = MemoryManager(
                    config=self._settings.memory,
                    bus=self._bus,
                    memory_dir=memory_dir,
                )
                await self._memory_mgr.initialize()
                log.info("MemoryManager initialised.")
            except Exception as exc:
                log.warning(
                    "MemoryManager failed to initialise (non-fatal): {exc}. "
                    "Running without memory.",
                    exc=exc,
                )
                self._memory_mgr = None

        # ── 7. Vision Engine (Milestone 9) ───────────────────────────────
        if self._settings.vision.enabled:
            try:
                from spidy.vision.manager import VisionManager
                self._vision_mgr = VisionManager(
                    config=self._settings.vision,
                    bus=self._bus,
                )
                await self._vision_mgr.initialize()
                log.info("VisionManager initialised.")
            except Exception as exc:
                log.warning(
                    "VisionManager failed to initialise (non-fatal): {exc}. "
                    "Running without vision.",
                    exc=exc,
                )
                self._vision_mgr = None

        # ── 8. Knowledge Engine (Milestone 10) ───────────────────────
        if getattr(self._settings.knowledge, "enabled", True):
            try:
                from spidy.knowledge.manager import KnowledgeManager
                knowledge_dir = self._settings.paths.resolve("data_dir") / "knowledge"
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                self._knowledge_mgr = KnowledgeManager(
                    config=self._settings.knowledge,
                    bus=self._bus,
                    knowledge_dir=knowledge_dir,
                )
                await self._knowledge_mgr.initialize()
                log.info("KnowledgeManager initialised.")
            except Exception as exc:
                log.warning(
                    "KnowledgeManager failed to initialise (non-fatal): {exc}. "
                    "Running without knowledge engine.",
                    exc=exc,
                )
                self._knowledge_mgr = None

        # ── 9. Learning Engine (Milestone 11) ────────────────────────
        if getattr(self._settings.learning, "enabled", True):
            try:
                from spidy.learning.manager import LearningManager
                memory_dir = self._settings.paths.resolve("memory_dir")
                memory_dir.mkdir(parents=True, exist_ok=True)
                self._learning_mgr = LearningManager(
                    config=self._settings.learning,
                    bus=self._bus,
                    learning_dir=memory_dir,
                )
                await self._learning_mgr.initialize()
                log.info("LearningManager initialised.")
            except Exception as exc:
                log.warning(
                    "LearningManager failed to initialise (non-fatal): {exc}. "
                    "Running without learning engine.",
                    exc=exc,
                )
                self._learning_mgr = None

    async def _run_ollama_health_check(self) -> None:
        """
        Run the Ollama startup health check and print a diagnostic banner.

        Called immediately after configuration is loaded so the user sees
        the report before any other module starts.

        Behaviour
        ---------
        - Always prints the report banner (never silent).
        - In text mode: if the model is missing, offers an interactive pull.
        - In voice/UI mode: logs a warning if issues are detected (no stdin).
        - Never blocks startup — issues are advisory, not fatal.
        """
        assert self._settings is not None
        provider = getattr(self._settings.reasoning, "provider", "ollama").lower()
        if provider != "ollama":
            # Skip Ollama-specific checks for cloud providers
            return

        try:
            from spidy.llm.health import (
                run_health_check_async,
                print_health_report,
                maybe_pull_model,
            )
        except ImportError:
            # Should never happen — health.py is stdlib-only
            log.warning("Could not import health check module.")
            return

        report = await run_health_check_async(self._settings.reasoning)
        print_health_report(report)

        if report.all_ok:
            return

        if not report.server_running:
            log.warning(
                "Ollama server is not running. "
                "Brain will be unavailable until 'ollama serve' is started."
            )
            return

        if not report.model_available:
            if self._text_mode:
                # Text mode: terminal is available — offer auto-pull
                pulled = maybe_pull_model(
                    model=self._settings.reasoning.model,
                    base_url=self._settings.reasoning.base_url,
                )
                if pulled:
                    log.info(
                        "Model '{m}' pulled successfully.",
                        m=self._settings.reasoning.model,
                    )
            else:
                log.warning(
                    "Model '{m}' is not installed. "
                    "Run 'ollama pull {m}' then restart Spidy.",
                    m=self._settings.reasoning.model,
                )

    async def _run(self) -> None:
        """Step 2: Start all services and enter the main event loop."""
        self._transition(LifecycleState.RUNNING)
        loop = asyncio.get_running_loop()

        # ── Start VoiceEngine (skipped in text mode) ──────────────────────
        if not self._text_mode and self._settings and self._settings.voice.wake_word.enabled:
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
                from spidy.skills.builtin import register_builtin_skills
                from spidy.skills.desktop import register_desktop_skills

                skill_registry = SkillRegistry()
                llm_client = LLMClientFactory.build(self._settings.reasoning)

                # Register built-in skills
                register_builtin_skills(
                    skill_registry,
                    config=self._settings.skills,
                    bus=self._bus,
                    notes_file=getattr(self._settings.skills, "notes_file", "notes.jsonl"),
                )

                # Register desktop & file agent skills (Milestone 6)
                register_desktop_skills(
                    skill_registry,
                    config=self._settings.skills,
                    bus=self._bus,
                )

                # Register browser agent skills (Milestone 7)
                from spidy.skills.browser import register_browser_skills
                register_browser_skills(
                    skill_registry,
                    config=self._settings.skills,
                    bus=self._bus,
                )

                self._brain = Brain(
                    bus=self._bus,
                    config=self._settings.reasoning,
                    skill_registry=skill_registry,
                    llm_client=llm_client,
                    memory=self._memory_mgr,
                    vision=self._vision_mgr,
                    knowledge=self._knowledge_mgr,
                    learning=self._learning_mgr,
                    user_name=self._settings.app.user_name,
                )
                await self._brain.start()
                log.info("Brain running. Ready to process utterances.")

                # ── Start Brain↔UI bridge ─────────────────────────────────
                # Translates brain.* events → ui.* events so the overlay
                # chat view receives conversation turns. Active in all modes.
                from spidy.ui.brain_bridge import BrainUIBridge
                self._brain_ui_bridge = BrainUIBridge(bus=self._bus)
                self._brain_ui_bridge.start()
                log.info("BrainUIBridge started. Overlay will now display chat.")

                # ── Start Plugin Manager (Milestone 12) ───────────────────
                if self._settings.plugins.enabled:
                    try:
                        from spidy.plugins.manager import PluginManager
                        plugins_dir = Path(
                            getattr(self._settings.paths, "plugins_dir", "plugins")
                        ).resolve()
                        plugins_dir.mkdir(parents=True, exist_ok=True)
                        self._plugin_mgr = PluginManager(
                            config=self._settings.plugins,
                            bus=self._bus,
                            skill_registry=skill_registry,
                            plugins_dir=plugins_dir,
                        )
                        await self._plugin_mgr.initialize()
                        log.info("PluginManager running.")
                    except Exception as exc:
                        log.warning(
                            "PluginManager failed to start (non-fatal): {exc}. "
                            "Running without plugins.",
                            exc=exc,
                        )
                        self._plugin_mgr = None

            except Exception as exc:
                log.warning(
                    "Brain failed to start (non-fatal): {exc}. Running without Brain.",
                    exc=exc,
                )
                self._brain = None

        # ── Start Qt Overlay UI (Milestone 5) ─────────────────────────────
        if self._settings and self._settings.ui.enabled:
            self._start_ui_thread(loop)

        # Announce to all modules that we are live
        await self._bus.publish(SpidyStartedEvent())

        if self._text_mode:
            log.info(
                "Spidy is running in TEXT MODE. "
                "Type a command and press Enter. Type 'quit' or 'exit' to stop."
            )
            await self._run_text_repl()
        else:
            log.info("Spidy is running. Waiting for shutdown signal...")
            # Block until shutdown is requested
            await self._shutdown_event.wait()

    async def _run_text_repl(self) -> None:
        """
        Interactive text REPL for testing without voice hardware.

        Reads lines from stdin asynchronously and forwards them to
        Brain.process(). Exits on 'quit', 'exit', or CTRL+C / CTRL+D.
        """
        loop = asyncio.get_running_loop()
        print("\n" + "=" * 60)
        print("  Spidy Text REPL — type a command, press Enter")
        print("  Type 'quit' or 'exit' to stop.")
        print("=" * 60)

        while not self._shutdown_event.is_set():
            try:
                # run_in_executor keeps the asyncio loop alive while blocking on input
                line: str = await loop.run_in_executor(None, self._prompt_user)
            except EOFError:
                # CTRL+D
                break

            line = line.strip()
            if not line:
                continue

            if line.lower() in {"quit", "exit", "bye", "stop"}:
                break

            if self._brain is not None:
                try:
                    response = await self._brain.process(line)
                    if response:
                        print(f"\n  Spidy: {response}\n")
                    else:
                        print("  [Spidy: (no response)]\n")
                except Exception as exc:
                    log.error("Text REPL: brain error: {exc}", exc=exc)
                    print(f"  [Error: {exc}]\n")
            else:
                print("  [Brain is not running — cannot process input]\n")

        await self.request_shutdown()

    @staticmethod
    def _prompt_user() -> str:
        """Blocking stdin read — called from thread executor."""
        try:
            return input("You: ")
        except EOFError:
            raise

    def _start_ui_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the Qt overlay UI in a dedicated daemon thread.

        Qt requires its own event loop, which cannot share the asyncio loop.
        We pass the asyncio loop to SpidyApp so it can publish events back
        into asyncio thread-safely.
        """
        assert self._settings is not None
        assert self._bus is not None

        def _run_qt() -> None:
            try:
                from spidy.ui.app import SpidyApp
                self._ui_app = SpidyApp(
                    bus=self._bus,
                    config=self._settings,
                    loop=loop,
                )
                self._ui_app.start()
                exit_code = self._ui_app.run()
                log.info("Qt UI exited with code {code}.", code=exit_code)
            except ImportError as exc:
                log.warning(
                    "Qt UI not available (missing deps): {exc}. "
                    "Running without overlay.",
                    exc=exc,
                )
            except Exception as exc:
                log.warning(
                    "Qt UI failed to start (non-fatal): {exc}. "
                    "Running without overlay.",
                    exc=exc,
                )

        self._ui_thread = threading.Thread(
            target=_run_qt,
            name="SpidyQtUI",
            daemon=True,  # Dies automatically when the main process exits
        )
        self._ui_thread.start()
        log.info("Qt overlay UI thread started.")

    async def _stop(self) -> None:
        """Step 3: Graceful shutdown sequence."""
        if self._state in (LifecycleState.STOPPED, LifecycleState.STOPPING):
            return  # Already shutting down

        # Tear down plugins FIRST so their skills are unregistered cleanly
        if self._plugin_mgr is not None:
            try:
                await self._plugin_mgr.teardown()
            except Exception as exc:  # noqa: BLE001
                log.debug("PluginManager teardown error (non-fatal): {exc}", exc=exc)

        self._transition(LifecycleState.STOPPING)
        log.info("Stopping Spidy...")

        # Stop Qt UI (signal quit from asyncio side)
        if self._ui_app is not None:
            try:
                self._ui_app.quit()
            except Exception as exc:
                log.debug("UI quit error (non-fatal): {exc}", exc=exc)

        # Stop Brain↔UI bridge first (it holds brain.* subscriptions)
        if self._brain_ui_bridge is not None:
            self._brain_ui_bridge.stop()

        # Stop Brain
        if self._brain is not None:
            await self._brain.stop()

        # Stop KnowledgeManager
        if self._knowledge_mgr is not None:
            await self._knowledge_mgr.close()

        # Stop LearningManager
        if self._learning_mgr is not None:
            await self._learning_mgr.close()

        # Stop VisionManager
        if self._vision_mgr is not None:
            await self._vision_mgr.close()

        # Stop MemoryManager
        if self._memory_mgr is not None:
            await self._memory_mgr.close()

        # Stop ObserverManager
        if self._observer_mgr is not None:
            await self._observer_mgr.stop()

        # Stop VoiceEngine last (releases audio resources)
        if self._voice_engine is not None:
            await self._voice_engine.stop()

        if self._bus:
            log.debug("Event bus stats: {stats}", stats=self._bus.stats)

        self._transition(LifecycleState.STOPPED)
        log.info(
            "Spidy stopped cleanly. Goodbye, {name}.",
            name=self._settings.app.user_name if self._settings else "",
        )

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
