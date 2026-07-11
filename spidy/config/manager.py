"""
Spidy Configuration Manager
============================
Loads, validates, and hot-reloads the master YAML configuration.

Architecture
------------
- Reads ``config/spidy_config.yaml`` at startup.
- Validates all values using Pydantic v2 models (type-safe, not just dicts).
- Provides a single global ``settings`` object accessible from anywhere.
- Supports environment variable overrides with the ``SPIDY_`` prefix.
- Watches the config file for changes and reloads automatically (optional).

Usage
-----
    from spidy.config import settings

    print(settings.app.user_name)
    print(settings.voice.stt.model)
    print(settings.ui.theme)

Design notes
------------
- We use nested Pydantic BaseModel classes (not BaseSettings) for the
  config tree, keeping it simple and not mixing config file + env vars in
  the same model.
- A thin SpidySettings wrapper handles the environment variable override
  layer via pydantic-settings.
- ConfigManager is a singleton: call ``ConfigManager.load()`` once at
  startup, then use ``settings`` everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# ─── Pydantic Config Models ───────────────────────────────────────────────────


class AppConfig(BaseModel):
    name: str = "Spidy"
    version: str = "0.1.0"
    user_name: str = "User"
    debug: bool = False
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v


class PathsConfig(BaseModel):
    data_dir: str = "%APPDATA%/Spidy"
    log_dir: str = "%APPDATA%/Spidy/logs"
    memory_dir: str = "%APPDATA%/Spidy/memory"
    models_dir: str = "assets/models"
    plugins_dir: str = "plugins"

    def resolve(self, key: str) -> Path:
        """Expand environment variables and return an absolute Path."""
        raw = getattr(self, key)
        return Path(os.path.expandvars(raw)).resolve()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "30 days"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{extra[module]:<30} | {message}"
    )
    colorize: bool = True
    backtrace: bool = True
    diagnose: bool = True


class WakeWordConfig(BaseModel):
    enabled: bool = True
    model: str = "hey_jarvis"
    threshold: float = 0.5
    chunk_size: int = 1280

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class STTConfig(BaseModel):
    model: str = "base.en"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    vad_filter: bool = True
    vad_threshold: float = 0.5


class TTSConfig(BaseModel):
    engine: str = "piper"
    voice: str = "en_US-ryan-high"
    speed: float = 1.0
    volume: float = 1.0


class AudioConfig(BaseModel):
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    channels: int = 1


class VoiceConfig(BaseModel):
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


class ReasoningConfig(BaseModel):
    """LLM provider and generation settings."""
    provider: str = "ollama"       # "ollama" | "openai" | "claude" | "gemini"
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: int = 30
    system_prompt_path: str = "config/system_prompt.txt"


class BrainConfig(BaseModel):
    """
    Configuration for the Brain Core (Milestone 3) and its future
    Lifelong AI Companion extensions.

    Companion feature flags
    -----------------------
    These flags are ``False`` by default and will be activated by future
    milestones when the corresponding implementations are ready.
    Setting them to ``True`` before implementation is available has no
    effect — the Brain will log a warning and continue without the feature.
    """

    # ── Intent Classifier ─────────────────────────────────────────────────
    # Classifier strategy: "heuristic" (M3) | "llm" (M4+)
    intent_classifier: str = "heuristic"
    # Minimum confidence to act on an intent without asking to clarify
    min_intent_confidence: float = 0.6

    # ── Conversation Window ───────────────────────────────────────────────
    # Maximum number of turns to keep in the rolling context window
    max_conversation_turns: int = 20

    # ── Decision Engine ───────────────────────────────────────────────────
    # Decision mode override: "auto" | "skill_only" | "llm_only"
    decision_mode: str = "auto"
    # Whether to route intents to registered skills
    tool_routing_enabled: bool = True

    # ── Companion Features (future milestones) ────────────────────────────
    # Milestone 6: Long-term memory, episodic recall, semantic search
    enable_memory: bool = False
    # Milestone 8+: Knowledge graph, RAG, document ingestion
    enable_knowledge: bool = False
    # Milestone 9+: Preference learning, habit detection, feedback loop
    enable_learning: bool = False

    # ── Multi-LLM Support (Milestone 4+) ─────────────────────────────────
    # Whether to enable optional web search (requires user permission)
    enable_web_search: bool = False


class ShortTermMemoryConfig(BaseModel):
    max_messages: int = 20


class LongTermMemoryConfig(BaseModel):
    db_filename: str = "spidy_memory.db"


class SemanticMemoryConfig(BaseModel):
    collection_name: str = "spidy_memories"
    embedding_model: str = "all-MiniLM-L6-v2"


class MemoryConfig(BaseModel):
    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)
    semantic: SemanticMemoryConfig = Field(default_factory=SemanticMemoryConfig)


class UIConfig(BaseModel):
    enabled: bool = True
    theme: str = "dark"
    opacity: float = 0.92
    position: str = "top-right"
    width: int = 420
    height: int = 600
    always_on_top: bool = True
    animate: bool = True
    hotkey: str = "ctrl+space"


class PermissionsConfig(BaseModel):
    policy_file: str = "permissions/policies/default_policy.yaml"
    require_confirmation_for: list[str] = Field(default_factory=list)
    audit_log_enabled: bool = True
    audit_log_file: str = "spidy_audit.log"


class PluginsConfig(BaseModel):
    enabled: bool = True
    auto_discover: bool = True
    sandbox: bool = True


class SystemConfig(BaseModel):
    poll_interval_seconds: int = 5
    cpu_alert_threshold: int = 90
    ram_alert_threshold: int = 85
    battery_alert_threshold: int = 20


class ContextConfig(BaseModel):
    """Configuration for the Milestone 2 Context Observer layer."""

    enabled: bool = True

    # Snapshot (aggregate event published periodically)
    snapshot_interval: float = 10.0   # seconds between snapshot events

    # Active Window
    window_poll_interval: float = 0.5  # seconds

    # Clipboard
    clipboard_enabled: bool = True
    clipboard_poll_interval: float = 1.0

    # Process monitor
    process_monitor_enabled: bool = True
    process_poll_interval: float = 3.0

    # System resources
    resource_monitor_enabled: bool = True
    resource_poll_interval: float = 5.0
    cpu_alert_threshold: float = 90.0
    ram_alert_threshold: float = 85.0
    battery_alert_threshold: float = 20.0

    # Downloads
    download_monitor_enabled: bool = True


    # Notifications
    notification_monitor_enabled: bool = True


class SpidyConfig(BaseModel):
    """
    Root configuration model.

    This is the single object that represents the entire spidy_config.yaml.
    All sub-models have sensible defaults so Spidy can run even with a
    minimal or missing config file.
    """

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)


# ─── ConfigManager ────────────────────────────────────────────────────────────


class ConfigManager:
    """
    Singleton manager that owns the loaded SpidyConfig.

    Usage
    -----
        from spidy.config.manager import ConfigManager

        # At startup:
        config_mgr = ConfigManager(config_path=Path("config/spidy_config.yaml"))
        config_mgr.load()

        # Anywhere:
        print(config_mgr.settings.app.user_name)
    """

    _DEFAULT_CONFIG_PATH = Path("config") / "spidy_config.yaml"

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or self._DEFAULT_CONFIG_PATH
        self._settings: SpidyConfig | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def load(self) -> SpidyConfig:
        """
        Load and validate the config file.

        - If the file does not exist, returns a SpidyConfig with all defaults
          and emits a warning.
        - If the YAML is invalid, raises ValueError with a clear message.
        - If a field fails Pydantic validation, raises ValueError.

        Returns
        -------
        SpidyConfig
            The fully validated configuration object.
        """
        raw: dict[str, Any] = {}

        if self._config_path.exists():
            log.debug("Loading config from {path}", path=self._config_path)
            try:
                with self._config_path.open(encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Failed to parse config file '{self._config_path}': {exc}"
                ) from exc
        else:
            log.warning(
                "Config file not found at '{path}'. Using all defaults.",
                path=self._config_path,
            )

        # Apply environment variable overrides BEFORE Pydantic validation.
        raw = self._apply_env_overrides(raw)

        try:
            self._settings = SpidyConfig(**raw)
        except Exception as exc:
            raise ValueError(f"Config validation failed: {exc}") from exc

        log.info(
            "Config loaded | user={user} | debug={debug}",
            user=self._settings.app.user_name,
            debug=self._settings.app.debug,
        )
        return self._settings

    @property
    def settings(self) -> SpidyConfig:
        """
        Return the currently loaded settings.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called yet.
        """
        if self._settings is None:
            raise RuntimeError(
                "ConfigManager.load() must be called before accessing settings."
            )
        return self._settings

    def reload(self) -> SpidyConfig:
        """
        Re-read the config file from disk and re-validate.

        Useful for hot-reload during development.
        """
        log.info("Reloading config from disk...")
        self._settings = None
        return self.load()

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Apply SPIDY_* environment variable overrides.

        Convention: SPIDY__APP__USER_NAME → raw["app"]["user_name"]
        Double underscore (__) separates nesting levels.
        """
        prefix = "SPIDY__"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("__")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
            log.debug("Env override applied: {key}={value}", key=key, value=value)
        return raw
