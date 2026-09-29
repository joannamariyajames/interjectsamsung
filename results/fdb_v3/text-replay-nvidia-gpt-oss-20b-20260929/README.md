# FDB-v3 text replay - NVIDIA-hosted openai/gpt-oss-20b (not the scored audio run)

What this is: `python -m app.fdb.replay` over all 100 recordings - each scenario's
spoken words (metadata `dialogue[0].user`, the input only, never the expected
calls) fed as text to the declared benchmark agent: the benchmark's own
instructions + `ARGUMENT_RULES`, the 12 official tools behind the BACKSPACE
adapter, the argument clean-up (`app/fdb/arguments.py`), and
`openai/gpt-oss-20b` on NVIDIA's free API catalog (temperature 0, seed 7,
reasoning effort low). STT and TTS are skipped, so this measures tool use and
reasoning only; latency does not apply.

Code: commit `5779f72` (see `config.json`). Scored with the official
`evaluate_pass_rate.py` / `evaluate_tool_calls.py` from Full-Duplex-Bench
`3e799c4`, unmodified, **without** the LLM judge (exact argument matching -
stricter than the official gpt-4o judge on wording such as "Vegas" vs
"Las Vegas").

| | Value |
|---|---|
| Recordings | 100 (0 errored) |
| **Strict pass rate** | **69.0%** (69/100) |
| Turn-take success | 99.0% |
| Tool selection accuracy | 97.0% |
| Argument accuracy | 78.1% |

| Domain | Pass rate | | Difficulty | Pass rate |
|---|---|---|---|---|
| finance / billing | 100.0% | | easy | 86.1% |
| e-commerce support | 75.9% | | medium | 61.8% |
| travel / identity | 60.0% | | hard | 56.7% |
| housing / location | 38.5% | | | |

For comparison, the earlier all-Groq configuration (`gpt-oss-120b`, no argument
clean-up) scored 52.0% on the same official report
([`../text-replay-gpt-oss-120b-20260928`](../text-replay-gpt-oss-120b-20260928)),
63.0% on the 81 recordings that completed before Groq's free daily cap.

How the model was chosen: on a 24-scenario sample, gpt-oss-20b and
`nvidia/nemotron-3-super-120b-a12b` both reached 83.3% with the argument
clean-up; gpt-oss-20b was kept for its faster, steadier responses (median 3.8 s
vs 5.0 s per recording, worst 22 s vs 35 s) and consistent argument types. On the
55 scenarios not used for tuning, gpt-oss-20b scored 65.5%, which is the honest
estimate this full replay confirms.

Files: `results/` (per-recording result files the evaluators read),
`*_pass_rate_report.json` / `*_evaluation_report.json` (official reports),
`agent_tool_calls.log` (every executed call), `replay.log`, `config.json`,
`pip-freeze.txt`.
