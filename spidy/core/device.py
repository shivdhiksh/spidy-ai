"""
DeviceManager — GPU/CPU Detection and Assignment
=================================================
Detects CUDA availability at startup and assigns compute devices to
each AI component based on VRAM budget and config preferences.

Design
------
- Called once in SpidyCore._initialise() before any AI module starts.
- Publishes a DeviceInfo to all modules via EventBus OR stores in a
  module-level singleton (chosen: singleton for simplicity, since this
  is read-only after startup).
- Gracefully falls back to CPU if CUDA is unavailable or VRAM is low.
- Logs VRAM budget clearly so Milestone 1 audio can be debugged easily.

VRAM Budget (RTX 3050 4GB):
  faster-whisper base.en  float16  ~145 MB  ← always loaded
  EasyOCR                 ResNet   ~400 MB  ← load on-demand
  Ollama LLM              q4       ~2.2 GB  ← managed by Ollama separately
  ─────────────────────────────────────────
  Spidy-managed total:             ~545 MB  (safe with 4 GB)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from spidy.logging.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """
    Immutable device configuration resolved at startup.

    Attributes
    ----------
    cuda_available:
        Whether a CUDA-capable GPU was detected.
    device:
        The device string to pass to PyTorch / faster-whisper.
        Either ``"cuda:0"`` or ``"cpu"``.
    provider:
        Short string for logging: ``"cuda"`` or ``"cpu"``.
    gpu_name:
        GPU model name, e.g. ``"NVIDIA GeForce RTX 3050 Laptop GPU"``.
        Empty string if no GPU.
    vram_bytes:
        Total VRAM in bytes. 0 if no GPU.
    vram_gb:
        Total VRAM in GB (rounded to 2 dp). Convenience attribute.
    whisper_compute_type:
        The compute type to pass to faster-whisper.
        ``"float16"`` on CUDA, ``"int8"`` on CPU.
    """
    cuda_available: bool
    device: str
    provider: str
    gpu_name: str
    vram_bytes: int
    whisper_compute_type: str

    @property
    def vram_gb(self) -> float:
        return round(self.vram_bytes / (1024 ** 3), 2)


class DeviceManager:
    """
    Singleton that owns the resolved DeviceInfo.

    Usage
    -----
        # At startup (once):
        dm = DeviceManager()
        dm.detect()

        # Everywhere else:
        device = DeviceManager.get().device  # "cuda:0" or "cpu"
    """

    _instance: ClassVar[DeviceInfo | None] = None

    def detect(self) -> DeviceInfo:
        """
        Detect CUDA availability and resolve the device configuration.

        Calls torch.cuda under the hood. If torch is not installed,
        gracefully falls back to CPU.

        Returns
        -------
        DeviceInfo
            The resolved device configuration. Also stored as singleton.
        """
        info = self._do_detect()
        DeviceManager._instance = info
        self._log_summary(info)
        return info

    @staticmethod
    def get() -> DeviceInfo:
        """
        Return the cached DeviceInfo.

        Raises
        ------
        RuntimeError
            If detect() has not been called yet.
        """
        if DeviceManager._instance is None:
            raise RuntimeError(
                "DeviceManager.detect() must be called during startup "
                "before accessing device info."
            )
        return DeviceManager._instance

    @staticmethod
    def _do_detect() -> DeviceInfo:
        """Core detection logic."""
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return DeviceInfo(
                    cuda_available=True,
                    device="cuda:0",
                    provider="cuda",
                    gpu_name=props.name,
                    vram_bytes=props.total_memory,
                    whisper_compute_type="float16",
                )
            else:
                return DeviceInfo(
                    cuda_available=False,
                    device="cpu",
                    provider="cpu",
                    gpu_name="",
                    vram_bytes=0,
                    whisper_compute_type="int8",
                )

        except ImportError:
            # torch not installed — fall back to CPU gracefully
            log.warning(
                "PyTorch not found. AI models will run on CPU. "
                "Install torch for GPU acceleration."
            )
            return DeviceInfo(
                cuda_available=False,
                device="cpu",
                provider="cpu",
                gpu_name="",
                vram_bytes=0,
                whisper_compute_type="int8",
            )

    @staticmethod
    def _log_summary(info: DeviceInfo) -> None:
        """Log a clear device summary at startup."""
        if info.cuda_available:
            log.info(
                "GPU detected: {gpu} | VRAM: {vram} GB | "
                "Whisper compute: {ct} | Device: {dev}",
                gpu=info.gpu_name,
                vram=info.vram_gb,
                ct=info.whisper_compute_type,
                dev=info.device,
            )
        else:
            log.info(
                "No GPU detected. All AI models will run on CPU. "
                "Whisper compute type: {ct}",
                ct=info.whisper_compute_type,
            )
