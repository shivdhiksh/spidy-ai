"""
Spidy — AI Desktop Companion
Root package initialiser.

CUDA Runtime DLL Patch
----------------------
On Windows, nvidia-cublas-cu12 and nvidia-cuda-runtime-cu12 install their
DLLs inside the Python site-packages tree (e.g. nvidia/cublas/bin/).
ctranslate2 loads these DLLs at C-extension import time via the Windows DLL
loader, which searches PATH.  If the nvidia bin directories are not on PATH
at that point, cublas64_12.dll / cudart64_12.dll are not found and Whisper
CUDA inference fails with RuntimeError even though the packages are installed.

This module patches os.environ['PATH'] with the nvidia bin directories
*before* any ctranslate2 / faster_whisper import can happen (those are
deferred to VoiceEngineFactory.build() at startup, well after this runs).

This is idempotent and silent — if the packages are absent, nothing changes.
"""

import os as _os
import sys as _sys

# ── NVIDIA pip-package DLL path injection (Windows only) ─────────────────
if _sys.platform == "win32":
    try:
        import site as _site

        _nvidia_pkgs = [
            "nvidia/cublas/bin",
            "nvidia/cuda_runtime/bin",
            "nvidia/cuda_nvrtc/bin",
        ]
        _injected: list[str] = []
        for _sp in _site.getsitepackages():
            for _pkg in _nvidia_pkgs:
                _dll_dir = _os.path.join(_sp, _pkg.replace("/", _os.sep))
                if _os.path.isdir(_dll_dir) and _dll_dir not in _os.environ.get("PATH", ""):
                    _injected.append(_dll_dir)

        if _injected:
            _os.environ["PATH"] = ";".join(_injected) + ";" + _os.environ.get("PATH", "")
            # Log is not available at this point — use print (captured by logger later)
            # This message appears only at process start, never during tests.
            # print(f"[Spidy] NVIDIA DLL paths injected: {len(_injected)} dir(s)")

        del _site, _nvidia_pkgs, _injected, _dll_dir, _sp, _pkg
    except Exception:
        pass  # Never crash at import time; ctranslate2 will fall back to CPU


__version__ = "0.1.0"
__author__ = "Shiva"
