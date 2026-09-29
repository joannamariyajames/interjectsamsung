"""Helpers for scripts/run_fdb_v3.sh - kept in Python so paths behave the same on
Linux, macOS and Windows (the official scripts write to Python's ``/tmp``, which
on Windows is ``<drive>:\\tmp``, not Git Bash's /tmp).

    python -m app.fdb.bench_utils subset <data_dir> <out_dir> <n>
    python -m app.fdb.bench_utils reset-telemetry <archive_dir>
    python -m app.fdb.bench_utils collect <data_dir> <label> <run_dir>
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from app.fdb import pipeline_config

TELEMETRY_FILES = (
    Path("/tmp/agent_tool_calls.log"),
    Path("/tmp/agent_heartbeat.log"),
    Path("/tmp/agent_llm_served.log"),  # which model answered each request (FDB_LLM_FALLBACK)
)


def llm_served_counts(log: Path) -> dict[str, int]:
    """Requests answered per model, from the runner's served-model log."""
    counts: dict[str, int] = {}
    if not log.exists():
        return counts
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            model = json.loads(line)["model"]
        except (ValueError, KeyError, TypeError):
            continue
        counts[model] = counts.get(model, 0) + 1
    return counts


def subset(data_dir: str, out_dir: str, n: str) -> None:
    """Copy the first recording of the first n scenarios (sorted) for a smoke run."""
    src, dst = Path(data_dir), Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    seen: set[str] = set()
    for d in sorted(p for p in src.iterdir() if (p / "metadata.json").exists()):
        scenario_id = json.loads((d / "metadata.json").read_text(encoding="utf-8"))["id"]
        if scenario_id in seen:
            continue
        seen.add(scenario_id)
        target = dst / d.name
        target.mkdir()
        for name in ("input.wav", "metadata.json"):
            shutil.copy2(d / name, target / name)
        if len(seen) == int(n):
            break
    print(f"subset: {len(seen)} scenario(s) in {dst}")


def reset_telemetry(archive_dir: str) -> None:
    """Move old agent telemetry aside so this run's log holds only this run."""
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    for f in TELEMETRY_FILES:
        if f.exists():
            shutil.move(str(f), archive / f"{f.stem}.before-run{f.suffix}")
    TELEMETRY_FILES[0].parent.mkdir(parents=True, exist_ok=True)
    print(f"telemetry reset: {TELEMETRY_FILES[0].resolve()}")


def _git_rev(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def collect(data_dir: str, label: str, run_dir: str) -> None:
    """Copy per-scenario results, telemetry and the run configuration into run_dir."""
    data, run = Path(data_dir), Path(run_dir)
    results = run / "results"
    results.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in data.glob(f"*/result_{label}.json"):
        shutil.copy2(f, results / f"{f.parent.name}.json")
        copied += 1
    for f in TELEMETRY_FILES:
        if f.exists():
            shutil.copy2(f, run / f.name)
    repo = Path(__file__).resolve().parents[3]
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (run / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    config = {
        "label": label,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "interject_commit": _git_rev(repo),
        "full_duplex_bench_commit": _git_rev(data.parents[1]) if len(data.parents) > 1 else "unknown",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **pipeline_config.describe(),
        "fdb_max_tool_steps": os.getenv("FDB_MAX_TOOL_STEPS", "5"),
        "llm_judge": "gpt-4o (--use-llm)" if os.getenv("FDB_JUDGE") == "1" else "none (exact argument matching)",
        "scenarios_collected": copied,
    }
    served = llm_served_counts(TELEMETRY_FILES[2])
    if served:
        config["llm_requests_served"] = served
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"collected {copied} result file(s) into {run}")


if __name__ == "__main__":
    commands = {"subset": subset, "reset-telemetry": reset_telemetry, "collect": collect}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](*sys.argv[2:])
