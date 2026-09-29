# FDB-v3 text replay - Groq openai/gpt-oss-120b (not the scored audio run)

What this is: `python -m app.fdb.replay` over all 100 recordings - each scenario's
spoken words (metadata `dialogue[0].user`, the input only) fed as text to the
benchmark agent (benchmark instructions + ARGUMENT_RULES, the 12 official tools
behind the BACKSPACE adapter, Groq `openai/gpt-oss-120b`, temperature 0). STT and
TTS are skipped, so this measures tool use and reasoning only; latency and
response-quality metrics do not apply.

Scored with the official `evaluate_pass_rate.py` / `evaluate_tool_calls.py`,
unmodified, **without** the LLM judge (exact argument matching - stricter than the
official gpt-4o judge on formatting such as dates).

| | Recordings | Passed | Strict pass rate |
|---|---|---|---|
| All 100 - official report (errored count as failures) | 100 | 52 | 52.0% |
| Completed without error (official per-recording check) | 81 | 51 | 63.0% |

One errored recording still passes in the official report: the cap was hit on the
agent's final spoken reply, after its tool calls had run correctly.

19 recordings (the last in alphabetical order, mostly travel and housing) did not
run: Groq's free tier caps `openai/gpt-oss-120b` at 200k tokens per day and the
cap was reached. They are recorded as errors in `results/*.json` and
`replay.log`, and counted as failures above.

Files: `*_pass_rate_report.json`, `*_evaluation_report.json` (official reports),
`results/` (per-recording result files), `tool_calls.log` (the tool-call
telemetry), `summary_completed_only.json`, `config.json`, `pip-freeze.txt`.
