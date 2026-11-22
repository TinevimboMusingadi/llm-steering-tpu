"""Device helpers: pick TPU when torch_tpu is installed, otherwise CPU; sync before timing; switch eager mode."""
from __future__ import annotations

import os
import time

import torch


def tpu_available() -> bool:
    try:
        import torch_tpu  # noqa: F401
    except Exception:
        return False
    return True


def get_device(name: str | None = None) -> torch.device:
    """`name` > $STEER_DEVICE > tpu if available > cpu."""
    name = name or os.environ.get("STEER_DEVICE") or ("tpu" if tpu_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device | str | None = None) -> None:
    """Block until queued device work is done. TPU execution is async: always sync before reading a clock."""
    kind = torch.device(device).type if device is not None else None
    if kind == "cpu":
        return
    try:
        torch.accelerator.synchronize()
    except Exception:
        if kind == "cuda":
            torch.cuda.synchronize()


def set_eager_mode(mode: str) -> str:
    """mode: "strict" (one op at a time, the default) or "defer" (DeferAndFuse: queue and fuse ops into bigger graphs).
    Returns what was actually set ("n/a" off-TPU)."""
    if not tpu_available():
        return "n/a"
    import torch_tpu  # noqa: F401
    em = torch.tpu.EagerMode
    torch.tpu.set_eager_mode(em.DEFER_AND_FUSE if mode == "defer" else em.DEFER_NEVER)
    return mode


class Timer:
    """with Timer(device) as t: ...  -> t.seconds, synced on both ends."""

    def __init__(self, device=None):
        self.device = device
        self.seconds = float("nan")

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self._t0
        return False


def env_stamp() -> dict:
    """Versions and settings, saved next to every result."""
    import platform

    import transformers

    stamp = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "defer_and_fuse_env": os.environ.get("TORCH_TPU_DEFER_AND_FUSE", ""),
    }
    try:
        import torch_tpu

        stamp["torch_tpu"] = getattr(torch_tpu, "__version__", "?")
    except Exception:
        stamp["torch_tpu"] = None
    return stamp
