from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys
from typing import Any


class QmbedError(RuntimeError):
    pass


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libqmbed_capi.dylib"
    if os.name == "nt":
        return "qmbed_capi.dll"
    return "libqmbed_capi.so"


def _library_path() -> Path:
    configured = os.environ.get("QMBED_LIBRARY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    repository = Path(__file__).resolve().parents[4]
    for profile in ("release", "debug"):
        candidate = (
            repository
            / "bindings"
            / "capi"
            / "target"
            / profile
            / _library_name()
        )
        if candidate.exists():
            return candidate
    raise QmbedError(
        "QMBED native library not found; set QMBED_LIBRARY_PATH or build "
        "bindings/capi with cargo"
    )


def run(request: dict[str, Any]) -> dict[str, Any]:
    library = ctypes.CDLL(str(_library_path()))
    library.qmbed_run_json.argtypes = [ctypes.c_char_p]
    library.qmbed_run_json.restype = ctypes.c_void_p
    library.qmbed_string_free.argtypes = [ctypes.c_void_p]
    library.qmbed_string_free.restype = None
    encoded = json.dumps(request, separators=(",", ":")).encode()
    pointer = library.qmbed_run_json(encoded)
    if not pointer:
        raise QmbedError("QMBED returned a null response")
    try:
        response = json.loads(ctypes.string_at(pointer))
    finally:
        library.qmbed_string_free(pointer)
    if response.get("status") != "ok":
        raise QmbedError(response.get("error", "unknown QMBED error"))
    return response["result"]
