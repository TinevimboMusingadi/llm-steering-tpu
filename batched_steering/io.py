"""Results on disk: results/<experiment>/<run>.json (+ samples.jsonl), with an env stamp. Checkpoint-friendly."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .device import env_stamp

ROOT = Path(os.environ.get("STEER_RESULTS", Path(__file__).resolve().parents[1] / "results"))


def run_dir(experiment: str) -> Path:
    d = ROOT / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def r3(x):
    return round(float(x), 4)


def save_json(experiment: str, name: str, data: dict, stamp: bool = True) -> Path:
    if stamp:
        data = {"env": env_stamp(), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    p = run_dir(experiment) / f"{name}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def load_json(experiment: str, name: str):
    p = ROOT / experiment / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def append_samples(experiment: str, rows: list[dict], name: str = "samples") -> Path:
    p = run_dir(experiment) / f"{name}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p
