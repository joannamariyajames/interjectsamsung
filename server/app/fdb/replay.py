"""Offline text replay of FDB-v3 through the benchmark agent - no LiveKit, no audio.

Feeds each scenario's spoken words - ``metadata.json`` ``dialogue[0]["user"]``,
the input only, never the expected calls - to the same agent the LiveKit runner
builds: the benchmark's VoiceAgent instructions, the 12 official tools behind
the BACKSPACE adapter, and the LLM of the configured cascaded pipeline
(``LK_PROVIDER``: ``nvidia`` by default, or ``groq``). Writes
``result_<name>.json`` in the schema the official evaluators read, so
``evaluate_tool_calls.py`` / ``evaluate_pass_rate.py`` score it unchanged.

STT and TTS are skipped, so this measures tool use and reasoning - not ASR
errors or latency. It is a fast, LiveKit-free way to iterate; the scored result
is still the real audio run (``scripts/run_fdb_v3``).

    python -m app.fdb.replay --examples ecommerce_09,finance_03
    python -m app.fdb.replay --limit 20 --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from app.fdb import pipeline_config
from app.fdb import runner as fdb_runner


def _scenario_dirs(
    data_dir: Path, examples: set[str] | None, limit: int | None, one_per_scenario: bool = False
) -> list[Path]:
    dirs = sorted(p for p in data_dir.iterdir() if (p / "metadata.json").exists())
    if examples or one_per_scenario:
        seen: set[str] = set()
        kept = []
        for p in dirs:
            scenario_id = json.loads((p / "metadata.json").read_text(encoding="utf-8"))["id"]
            if examples and scenario_id not in examples:
                continue
            if one_per_scenario and scenario_id in seen:
                continue
            seen.add(scenario_id)
            kept.append(p)
        dirs = kept
    return dirs[:limit] if limit else dirs


def _assistant_text(result: Any) -> str:
    parts = []
    for event in getattr(result, "events", []):
        item = getattr(event, "item", None)
        if getattr(event, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            text = getattr(item, "text_content", None)
            if text:
                parts.append(text)
    return " ".join(parts)


async def replay_one(scenario_dir: Path, name: str, telemetry_path: Path, timeout_s: float) -> dict[str, Any]:
    meta = json.loads((scenario_dir / "metadata.json").read_text(encoding="utf-8"))
    example_id = meta["id"]
    user_text = meta["dialogue"][0]["user"]
    room = f"replay-{example_id}-{uuid.uuid4().hex[:6]}"

    ctx = fdb_runner.create_fdb_runner_context(
        room,
        components={"llm": fdb_runner.build_llm()},
        nullable_optionals=True,
        telemetry_path=str(telemetry_path),
    )
    ctx.runtime.configure(token_delay_ms=0)
    started = time.time()
    transcript, error = "", None
    try:
        # Mirrors the LiveKit entrypoint: the final transcript also reaches the
        # BACKSPACE runtime, which tracks facts alongside the voice model.
        await ctx.runtime.on_final(user_text)
        async with ctx.session as session:
            await session.start(fdb_runner.InterjectVoiceAgent())
            result = await asyncio.wait_for(session.run(user_input=user_text), timeout_s)
            transcript = _assistant_text(result)
    except Exception as exc:  # noqa: BLE001 - recorded in the result, scenario marked as errored
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await ctx.runtime.shutdown()

    calls = []
    if telemetry_path.exists():
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("room") == room:
                calls.append(record["call"])

    result = {
        "example_id": example_id,
        "provider": name,
        "mode": "text_replay",
        **{k: v for k, v in pipeline_config.describe().items() if k.startswith("llm_")},
        "room_name": room,
        "actual_tool_calls": calls,
        "transcript": transcript,
        "error": error,
        "duration_s": round(time.time() - started, 2),
    }
    (scenario_dir / f"result_{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


async def main_async(args: argparse.Namespace) -> int:
    # The declared cascaded provider only: no hidden Gemini calls from any background component.
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(key, None)
    if os.environ.get("LK_PROVIDER", "").strip().lower() not in fdb_runner.CASCADED_PROVIDERS:
        os.environ["LK_PROVIDER"] = "nvidia"
    args.name = args.name or f"interject_{os.environ['LK_PROVIDER']}_replay"
    data_dir = Path(args.data_dir) if args.data_dir else fdb_runner._FDB_V3_DIR / "fdb_v3_data_released"
    examples = set(args.examples.split(",")) if args.examples else None
    # Default outside both repositories: the log is scratch output, not a result.
    telemetry_path = Path(args.telemetry) if args.telemetry else Path(tempfile.gettempdir()) / f"interject_replay_{args.name}.log"

    todo = []
    for d in _scenario_dirs(data_dir, examples, args.limit, args.one_per_scenario):
        if (d / f"result_{args.name}.json").exists() and not args.force:
            continue
        todo.append(d)
    print(f"replaying {len(todo)} scenario(s) as result_{args.name}.json (telemetry: {telemetry_path})")

    errors = 0
    for i, d in enumerate(todo, 1):
        if i > 1 and args.pace:
            await asyncio.sleep(args.pace)  # stay under the provider's per-minute token limit
        r = await replay_one(d, args.name, telemetry_path, args.timeout)
        errors += bool(r["error"])
        calls = ", ".join(f"{c['function']}({json.dumps(c['args'], ensure_ascii=False)})" for c in r["actual_tool_calls"])
        status = f"ERROR {r['error'][:120]}" if r["error"] else calls or "(no tool call)"
        print(f"[{i}/{len(todo)}] {r['example_id']:<22} {r['duration_s']:>5}s  {status}", flush=True)
    print(f"done: {len(todo) - errors} ok, {errors} errored")
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None, help="result_<name>.json and --provider for the evaluators "
                        "(default interject_<provider>_replay)")
    parser.add_argument("--data-dir", default=None, help="fdb_v3_data_released directory")
    parser.add_argument("--examples", default=None, help="comma-separated scenario ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per scenario")
    parser.add_argument("--telemetry", default=None, help="tool-call log (defaults to the system temp dir)")
    parser.add_argument("--one-per-scenario", action="store_true", help="first recording of each scenario only")
    parser.add_argument("--pace", type=float, default=20.0, help="seconds between scenarios (free-tier token limits)")
    raise SystemExit(asyncio.run(main_async(parser.parse_args())))


if __name__ == "__main__":
    main()
