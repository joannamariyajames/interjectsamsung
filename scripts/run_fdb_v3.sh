#!/usr/bin/env bash
# One-command Full-Duplex-Bench v3 run for Interject: install, configure, run the
# agent, run the official inference over all 100 recordings, evaluate, collect.
#
#   export LIVEKIT_URL=wss://<project>.livekit.cloud LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
#   export GROQ_API_KEY=gsk_...          # the agent's provider (Groq: Whisper + LLM + Orpheus)
#   export OPENAI_API_KEY=sk-...         # only with --judge: the official gpt-4o LLM judge (--use-llm)
#   ./scripts/run_fdb_v3.sh              # full run
#   ./scripts/run_fdb_v3.sh --subset 3   # smoke run on 3 scenarios
#
# Options: --subset N   --label NAME (default interject_groq)   --skip-install
#          --judge   score with the official gpt-4o judge (paid, needs OPENAI_API_KEY)
# Declared agent: LK_PROVIDER=groq (default) - a LiveKit cascaded voice agent:
#   Silero VAD -> Groq whisper-large-v3-turbo -> Groq openai/gpt-oss-120b (tools)
#   -> Groq canopylabs/orpheus-v1-english, tools gated by the BACKSPACE adapter.
# Results land in results/fdb_v3/<label>-<timestamp>/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/server"
FDB_DIR="${FDB_DIR:-$(dirname "$ROOT")/Full-Duplex-Bench}"   # the runner expects a sibling checkout
FDB_REPO="https://github.com/DanielLin94144/Full-Duplex-Bench.git"
FDB_COMMIT="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
FDB_DATA_GDRIVE="https://drive.google.com/file/d/1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz/view?usp=sharing"
LABEL="interject_groq"
SUBSET=""
SKIP_INSTALL=0
USE_JUDGE=0
export LK_PROVIDER="${LK_PROVIDER:-groq}"
export PYTHONIOENCODING=utf-8   # the official scripts print emoji

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) SUBSET="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --judge) USE_JUDGE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# -- 0. preflight -----------------------------------------------------------
for v in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
  [[ -n "${!v:-}" ]] || die "$v is not set (LiveKit Cloud credentials, see header)"
done
if [[ "$LK_PROVIDER" == groq* ]]; then
  [[ -n "${GROQ_API_KEY:-}" ]] || die "GROQ_API_KEY is not set"
fi
PY_BOOT="${PYTHON:-}"
if [[ -z "$PY_BOOT" ]]; then
  for c in python3.11 python3 python; do command -v "$c" >/dev/null 2>&1 && { PY_BOOT="$c"; break; }; done
fi
[[ -n "$PY_BOOT" ]] || die "Python 3.11 not found (set PYTHON=/path/to/python3.11)"
command -v git >/dev/null || die "git is required"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ROOT/results/fdb_v3/${LABEL}-${STAMP}"
mkdir -p "$RUN_DIR"
log "run directory: $RUN_DIR"

# -- 1. Full-Duplex-Bench checkout (pinned) and data ---------------------------
if [[ ! -d "$FDB_DIR/v3" ]]; then
  log "cloning Full-Duplex-Bench @ $FDB_COMMIT into $FDB_DIR"
  git clone -q "$FDB_REPO" "$FDB_DIR"
  git -C "$FDB_DIR" checkout -q "$FDB_COMMIT"
fi
V3="$FDB_DIR/v3"
export FDB_V3_DIR="$V3"   # tells the agent where the official tools live

# -- 2. Python environment ------------------------------------------------------
VENV="$SERVER/.venv-fdb"
if [[ ! -d "$VENV" ]]; then
  log "creating $VENV with $PY_BOOT"
  "$PY_BOOT" -m venv "$VENV"
fi
if [[ -x "$VENV/bin/python" ]]; then PY="$VENV/bin/python"; BIN="$VENV/bin"; else PY="$VENV/Scripts/python.exe"; BIN="$VENV/Scripts"; fi
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  log "installing pinned dependencies (server/requirements-fdb.txt)"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$SERVER/requirements-fdb.txt"
fi
export PATH="$BIN:$PATH"

if ! command -v ffmpeg >/dev/null 2>&1; then
  log "ffmpeg not on PATH: installing a bundled binary (imageio-ffmpeg)"
  "$PY" -m pip install -q imageio-ffmpeg==0.6.0
  FF="$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  if [[ "$BIN" == *Scripts ]]; then cp "$FF" "$BIN/ffmpeg.exe"; else ln -sf "$FF" "$BIN/ffmpeg"; fi
fi

if [[ "$LK_PROVIDER" == groq* ]]; then
  log "checking Groq access (LLM, STT, TTS)"
  "$PY" - <<'PYCHECK' || die "Groq preflight failed (see above)"
import os, sys, httpx
h = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
models = {m["id"] for m in httpx.get("https://api.groq.com/openai/v1/models", headers=h, timeout=30).json().get("data", [])}
need = [os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b"), os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")]
missing = [m for m in need if m not in models]
if missing:
    sys.exit(f"models not available to this Groq key: {missing}")
r = httpx.post("https://api.groq.com/openai/v1/audio/speech", headers=h, timeout=60,
               json={"model": need[2], "voice": os.getenv("GROQ_TTS_VOICE", "autumn"), "input": "Ready.", "response_format": "wav"})
if r.status_code != 200:
    sys.exit(f"TTS unavailable ({r.status_code}): {r.text[:300]}\n"
             "Orpheus needs its terms accepted once by the Groq org admin: "
             "https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english")
print("Groq OK:", ", ".join(need))
PYCHECK
fi

DATA="$V3/fdb_v3_data_released"
if [[ ! -d "$DATA" ]]; then
  log "downloading the benchmark data (Google Drive)"
  "$PY" -m gdown --fuzzy "$FDB_DATA_GDRIVE" -O "$FDB_DIR/fdb_v3_data_released.zip"
  "$PY" -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); [z.extract(m, sys.argv[2]) for m in z.namelist() if not m.startswith('__MACOSX') and not m.endswith('.DS_Store')]" \
    "$FDB_DIR/fdb_v3_data_released.zip" "$V3"
fi
[[ "$(ls -d "$DATA"/*/ | wc -l)" -ge 100 ]] || die "benchmark data incomplete in $DATA"

RUN_DATA="$DATA"
if [[ -n "$SUBSET" ]]; then
  RUN_DATA="$RUN_DIR/subset-data"   # inside the run folder: never write into the official checkout
  (cd "$SERVER" && "$PY" -m app.fdb.bench_utils subset "$DATA" "$RUN_DATA" "$SUBSET")
fi

# -- 3. start the agent --------------------------------------------------------
(cd "$SERVER" && "$PY" -m app.fdb.bench_utils reset-telemetry "$RUN_DIR/previous-telemetry")
log "starting the agent (LK_PROVIDER=$LK_PROVIDER)"
(cd "$SERVER" && exec "$PY" -m app.fdb.runner start) > "$RUN_DIR/agent.log" 2>&1 &
AGENT_PID=$!
cleanup() { kill "$AGENT_PID" 2>/dev/null || true; wait "$AGENT_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 90); do
  grep -q "registered worker" "$RUN_DIR/agent.log" && break
  kill -0 "$AGENT_PID" 2>/dev/null || { tail -30 "$RUN_DIR/agent.log"; die "agent exited during startup"; }
  sleep 2
done
grep -q "registered worker" "$RUN_DIR/agent.log" || { tail -30 "$RUN_DIR/agent.log"; die "agent did not register with LiveKit"; }
log "agent registered with LiveKit"

# -- 4. official inference -----------------------------------------------------
log "running official inference ($(ls -d "$RUN_DATA"/*/ | wc -l) recordings)"
# the official batch script, unchanged; CPU fallback for its ASR model only when no CUDA GPU exists
(cd "$V3" && "$PY" "$ROOT/scripts/fdb_official_inference.py" --provider "$LABEL" --root_dir "$RUN_DATA" --force) \
  2>&1 | tee "$RUN_DIR/inference.log"
cleanup
trap - EXIT

# -- 5. official evaluation ----------------------------------------------------
JUDGE=()
if [[ "$USE_JUDGE" -eq 1 ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || die "--judge needs OPENAI_API_KEY"
  JUDGE=(--use-llm)
fi
log "evaluating (${JUDGE[*]:-exact argument matching; pass --judge for the official gpt-4o judge})"
(cd "$V3" && "$PY" evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$RUN_DATA" \
    --provider "$LABEL" --output "$RUN_DIR/${LABEL}_evaluation_report.json" ${JUDGE[@]+"${JUDGE[@]}"}) 2>&1 | tee "$RUN_DIR/evaluate_tool_calls.log" \
  || log "evaluate_tool_calls.py failed - see evaluate_tool_calls.log (it cannot summarise a run where no scenario got a spoken response)"
(cd "$V3" && "$PY" evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$RUN_DATA" \
    --provider "$LABEL" --output "$RUN_DIR/${LABEL}_pass_rate_report.json" ${JUDGE[@]+"${JUDGE[@]}"}) 2>&1 | tee "$RUN_DIR/evaluate_pass_rate.log"
if [[ "$USE_JUDGE" -eq 1 ]]; then   # the official latency analysis also asks gpt-4o when the key information was spoken
  (cd "$V3" && "$PY" analyze_tool_latency.py --results-dir "$RUN_DATA" --provider "$LABEL") 2>&1 | tee "$RUN_DIR/latency.log" || true
  mv "$V3/${LABEL}_latency_report.json" "$RUN_DIR/" 2>/dev/null || true
else
  log "latency analysis skipped (needs --judge); first-response latency is in the evaluation report"
fi

# -- 6. collect ----------------------------------------------------------------
(cd "$SERVER" && FDB_JUDGE="$USE_JUDGE" "$PY" -m app.fdb.bench_utils collect "$RUN_DATA" "$LABEL" "$RUN_DIR")
# a smoke subset is a scratch copy of the audio: its results are collected above
[[ -n "$SUBSET" ]] && rm -rf "$RUN_DATA"
log "done - reports, per-scenario results, telemetry and config in $RUN_DIR"
