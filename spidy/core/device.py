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

import os
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from spidy.logging.logger import get_logger

log = get_logger(__name__)

_DLL_DIRS_ADDED = False


def ensure_cuda_dlls() -> list[str]:
    """
    Ensure NVIDIA CUDA runtime DLL directories from site-packages (e.g. nvidia-cublas-cu12,
    nvidia-cuda-runtime-cu12) are added to Python's DLL search paths on Windows.

    Returns
    -------
    list[str]
        List of directory paths that were successfully registered.
    """
    global _DLL_DIRS_ADDED
    added: list[str] = []
    if sys.platform != "win32":
        return added

    # Gather search candidate roots
    candidate_roots = list(site.getsitepackages())
    if site.getusersitepackages():
        candidate_roots.append(site.getusersitepackages())
    candidate_roots.append(sys.prefix)

    for root in candidate_roots:
        nv_dir = os.path.join(root, "nvidia")
        if os.path.isdir(nv_dir):
            try:
                for sub in os.listdir(nv_dir):
                    bin_dir = os.path.join(nv_dir, sub, "bin")
                    if os.path.isdir(bin_dir) and bin_dir not in added:
                        try:
                            os.add_dll_directory(bin_dir)
                            added.append(bin_dir)
                        except Exception:
                            pass
                        # Also prepend to PATH for subprocesses or legacy loaders
                        cur_path = os.environ.get("PATH", "")
                        if bin_dir not in cur_path:
                            os.environ["PATH"] = bin_dir + os.pathsep + cur_path
            except Exception:
                pass

    _DLL_DIRS_ADDED = True
    return added


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

        Registers NVIDIA CUDA runtime DLLs, probes ctranslate2 and PyTorch,
        and queries GPU properties. Gracefully falls back to CPU if no
        CUDA device or drivers are available.

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
        ensure_cuda_dlls()

        # 1. Probe PyTorch first if CUDA is available there
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
        except Exception:
            pass

        # 2. Probe ctranslate2 + NVIDIA driver (handles torch CPU build)
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                gpu_name, vram_bytes = DeviceManager._query_nvidia_smi()
                return DeviceInfo(
                    cuda_available=True,
                    device="cuda:0",
                    provider="cuda",
                    gpu_name=gpu_name or "NVIDIA CUDA GPU",
                    vram_bytes=vram_bytes or (4 * 1024 * 1024 * 1024),
                    whisper_compute_type="float16",
                )
        except Exception:
            pass

        # Fallback to CPU
        return DeviceInfo(
            cuda_available=False,
            device="cpu",
            provider="cpu",
            gpu_name="",
            vram_bytes=0,
            whisper_compute_type="int8",
        )

    @staticmethod
    def _query_nvidia_smi() -> tuple[str, int]:
        """Query GPU name and VRAM bytes via nvidia-smi if available."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,nounits,noheader"],
                encoding="utf-8",
                timeout=3,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            name = parts[0]
            vram_mb = float(parts[1]) if len(parts) > 1 else 0
            return name, int(vram_mb * 1024 * 1024)
        except Exception:
            return "", 0

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

