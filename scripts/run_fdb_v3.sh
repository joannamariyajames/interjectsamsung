#!/usr/bin/env bash
# One-command Full-Duplex-Bench v3 run for Interject: install, configure, run the
# agent, run the official inference over all 100 recordings, evaluate, collect.
#
# Keys (all free accounts), in the environment or in the repository-root .env:
#   LIVEKIT_URL=wss://<project>.livekit.cloud  LIVEKIT_API_KEY=...  LIVEKIT_API_SECRET=...
#   GROQ_API_KEY=gsk_...       # speech-to-text (Groq whisper-large-v3-turbo)
#   NVIDIA_API_KEY=nvapi-...   # the tool-calling LLM (build.nvidia.com free endpoint)
#   OPENAI_API_KEY=sk-...      # only with --judge: the official gpt-4o LLM judge (--use-llm)
#   ./scripts/run_fdb_v3.sh              # full run
#   ./scripts/run_fdb_v3.sh --subset 3   # smoke run on 3 scenarios
#
# Options: --subset N   --label NAME (default interject_<provider>)   --skip-install
#          --judge   score with the official gpt-4o judge (paid, needs OPENAI_API_KEY)
# Declared agent: LK_PROVIDER=nvidia (default) - a LiveKit cascaded voice agent:
#   Silero VAD -> Groq whisper-large-v3-turbo -> NVIDIA-hosted $FDB_LLM_MODEL
#   (default openai/gpt-oss-20b, tools) -> local Piper voice (en_US-ljspeech-medium),
#   tools gated by the BACKSPACE adapter. Backup LLM (FDB_LLM_FALLBACK=1, default):
#   Groq openai/gpt-oss-120b answers only a request NVIDIA fails or leaves silent for
#   FDB_LLM_ATTEMPT_TIMEOUT (10) s; the run log records which model served each one. LK_PROVIDER=groq keeps the all-Groq
#   pipeline (gpt-oss-120b + Orpheus), which needs a paid Groq tier for 100 recordings.
# Results land in results/fdb_v3/<label>-<timestamp>/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/server"
FDB_DIR="${FDB_DIR:-$(dirname "$ROOT")/Full-Duplex-Bench}"   # the runner expects a sibling checkout
FDB_REPO="https://github.com/DanielLin94144/Full-Duplex-Bench.git"
FDB_COMMIT="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
FDB_DATA_GDRIVE="https://drive.google.com/file/d/1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz/view?usp=sharing"
LABEL=""
SUBSET=""
SKIP_INSTALL=0
USE_JUDGE=0
export PYTHONIOENCODING=utf-8   # the official scripts print emoji

# Keys may live in the repository-root .env; variables already set in the
# environment win. Values are exported, never printed.
if [[ -f "$ROOT/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    key="${key#export }"; key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key:-}" ]] && continue
    value="${value%$'\r'}"; value="${value#\"}"; value="${value%\"}"; value="${value#\'}"; value="${value%\'}"
    export "$key=$value"
  done < "$ROOT/.env"
fi
export LK_PROVIDER="${LK_PROVIDER:-nvidia}"
export FDB_LLM_FALLBACK="${FDB_LLM_FALLBACK:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) SUBSET="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --judge) USE_JUDGE=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
LABEL="${LABEL:-interject_${LK_PROVIDER}}"
case "$LK_PROVIDER" in
  nvidia*) TTS_DEFAULT=piper ;;
  *) TTS_DEFAULT=orpheus ;;
esac
export FDB_TTS="${FDB_TTS:-$TTS_DEFAULT}"

# -- 0. preflight -----------------------------------------------------------
for v in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
  [[ -n "${!v:-}" ]] || die "$v is not set (LiveKit Cloud credentials, see header)"
done
if [[ "$LK_PROVIDER" == groq* || "$LK_PROVIDER" == nvidia* ]]; then
  [[ -n "${GROQ_API_KEY:-}" ]] || die "GROQ_API_KEY is not set (speech-to-text)"
fi
if [[ "$LK_PROVIDER" == nvidia* ]]; then
  KEY_ENV="${FDB_LLM_API_KEY_ENV:-NVIDIA_API_KEY}"
  [[ -n "${!KEY_ENV:-}" ]] || die "$KEY_ENV is not set (the LLM; free key at https://build.nvidia.com)"
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

if [[ "$LK_PROVIDER" == groq* || "$LK_PROVIDER" == nvidia* ]]; then
  log "checking provider access (LK_PROVIDER=$LK_PROVIDER, voice: $FDB_TTS)"
  "$PY" - <<'PYCHECK' || die "provider preflight failed (see above)"
import os, sys, httpx
provider, tts = os.environ["LK_PROVIDER"], os.environ["FDB_TTS"]
h = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
groq_models = {m["id"] for m in httpx.get("https://api.groq.com/openai/v1/models", headers=h, timeout=30).json().get("data", [])}
need = [os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")]
if provider.startswith("groq"):
    need.append(os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b"))
if tts == "orpheus":
    need.append(os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"))
missing = [m for m in need if m not in groq_models]
if missing:
    sys.exit(f"models not available to this Groq key: {missing}")
if tts == "orpheus":
    r = httpx.post("https://api.groq.com/openai/v1/audio/speech", headers=h, timeout=60,
                   json={"model": need[-1], "voice": os.getenv("GROQ_TTS_VOICE", "autumn"), "input": "Ready.", "response_format": "wav"})
    if r.status_code != 200:
        sys.exit(f"TTS unavailable ({r.status_code}): {r.text[:300]}\n"
                 "Orpheus needs its terms accepted once by the Groq org admin: "
                 "https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english")
print("Groq OK:", ", ".join(need))
fallback = os.getenv("FDB_LLM_FALLBACK", "0") == "1"
if fallback and provider.startswith("nvidia"):
    backup = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
    # the backup is optional: warn, never stop the run
    print(f"backup LLM: groq {backup}" if backup in groq_models
          else f"WARNING: backup LLM groq {backup} not available to this key - NVIDIA only")
if fallback and provider.startswith("groq") and not os.getenv(os.getenv("FDB_LLM_API_KEY_ENV", "NVIDIA_API_KEY")):
    print("WARNING: no NVIDIA_API_KEY - no backup LLM, Groq only")
if provider.startswith("nvidia"):
    base = os.getenv("FDB_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model = os.getenv("FDB_LLM_MODEL", "openai/gpt-oss-20b")
    key = os.environ[os.getenv("FDB_LLM_API_KEY_ENV", "NVIDIA_API_KEY")]
    # one tiny request: proves the key works and the model is served (a few tokens of a free quota);
    # a free endpoint can stall for a while, so a timeout is retried before the run is abandoned
    problem = ""
    for attempt in range(1, 4):
        try:
            r = httpx.post(f"{base}/chat/completions", timeout=60, headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "messages": [{"role": "user", "content": "Reply with the word ready."}],
                                 "max_tokens": 64, "temperature": 0})
        except httpx.TransportError as exc:
            problem = f"{type(exc).__name__} (attempt {attempt}/3)"
            print(f"LLM check: {problem}, retrying")
            continue
        if r.status_code == 200:
            print(f"LLM OK: {model} at {base}")
            break
        if r.status_code not in (429, 500, 502, 503, 504):
            sys.exit(f"LLM {model} at {base} unavailable ({r.status_code}): {r.text[:300]}")
        problem = f"HTTP {r.status_code} (attempt {attempt}/3)"
        print(f"LLM check: {problem}, retrying")
    else:
        sys.exit(f"LLM {model} at {base} did not answer: {problem}")
PYCHECK
fi
if [[ "$FDB_TTS" == piper ]]; then
  log "fetching the pinned local voice (Piper en_US-ljspeech-medium)"
  (cd "$SERVER" && "$PY" -m app.fdb.local_tts --download) || die "could not fetch the Piper voice"
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
