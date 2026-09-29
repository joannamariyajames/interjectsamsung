"""Member 4 FDB-v3 LiveKit Agent Runner with Backspace Adapter.

Reuses official Full-Duplex-Bench components as a read-only dependency.
Does NOT modify or copy official FDB files.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

# Resolve Full-Duplex-Bench/v3 path as read-only external dependency
_REPO_ROOT = Path(__file__).resolve().parents[3]  # interjectsamsung
# Full-Duplex-Bench is expected as a sibling checkout, but how many
# directory levels up that sibling sits depends on whether this repo was
# checked out directly (Full-Duplex-Bench next to "interjectsamsung") or
# nested one level deeper inside an extracted folder (e.g.
# "interjectsamsung-main/interjectsamsung", with Full-Duplex-Bench next to
# "interjectsamsung-main" instead). Try the direct-sibling location first
# (unchanged behaviour wherever that was already correct), then one level
# higher, and use whichever actually exists.
_FDB_V3_CANDIDATES = [
    *([Path(os.environ["FDB_V3_DIR"])] if os.environ.get("FDB_V3_DIR") else []),
    _REPO_ROOT.parent / "Full-Duplex-Bench" / "v3",
    _REPO_ROOT.parent.parent / "Full-Duplex-Bench" / "v3",
]
_FDB_V3_DIR = next((p for p in _FDB_V3_CANDIDATES if p.exists()), _FDB_V3_CANDIDATES[0])
if _FDB_V3_DIR.exists() and str(_FDB_V3_DIR) not in sys.path:
    sys.path.insert(0, str(_FDB_V3_DIR))

# Ensure interjectsamsung/server is in sys.path for app.backspace and app.fdb
_SERVER_DIR = _REPO_ROOT / "server"
if _SERVER_DIR.exists() and str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

try:
    from dotenv import load_dotenv

    # Same guard as app/livekit_worker.py: importing this module during a test run
    # (e.g. via a test that exercises resolve_realtime_model()) must never
    # reload real credentials from .env.local into os.environ - conftest.py
    # deliberately strips GEMINI_API_KEY once, at collection time, so tests
    # exercise MockProvider; an unconditional load_dotenv() here would load
    # them straight back (and, unlike a per-test monkeypatch, that load is
    # never reverted - it persists in os.environ for the rest of the
    # process), for every test that imports this module afterwards.
    _ENV_LOCAL = _REPO_ROOT / ".env.local"
    if (
        "pytest" not in sys.modules
        and "PYTEST_CURRENT_TEST" not in os.environ
        and "PYTEST_VERSION" not in os.environ
    ):
        if _ENV_LOCAL.exists():
            load_dotenv(_ENV_LOCAL)
        # The repository-root .env (LIVEKIT_* / GEMINI_API_KEY), which
        # app/config.py used to load for every process. Loaded after
        # .env.local, which therefore still takes precedence.
        load_dotenv()
except ImportError:
    pass

from app.backspace import BackspaceCore
from app.fdb.adapter import FDBBackspaceAdapter
from app.fdb.arguments import normalize_arguments

try:
    from livekit import agents, rtc
    from livekit.agents import Agent, AgentServer, AgentSession, APIConnectOptions, llm
    from livekit.agents.voice.agent_session import SessionConnectOptions
    from livekit.plugins import google
    from lk_agent_tool import (
        AssistantFnc,
        LatencyTracker,
        VoiceAgent,
        get_realtime_model,
        registry,
    )
except ImportError as err:
    # Allow module import even when livekit dependencies are not in environment
    logging.warning("LiveKit / FDB dependencies unavailable in runner: %s", err)
    agents = None
    rtc = None
    AgentServer = None
    AgentSession = None
    Agent = None
    llm = None
    google = None
    AssistantFnc = None
    VoiceAgent = None
    get_realtime_model = None
    LatencyTracker = None
    registry = None

# LiveKit requires plugins to be registered on the main thread, i.e. imported
# when the worker starts - importing them inside a job's entrypoint fails with
# "Plugins must be registered on the main thread" and the job never answers.
try:
    from livekit.plugins import groq as groq_plugin
    from livekit.plugins import openai as openai_plugin
    from livekit.plugins import silero
except ImportError:  # the cascaded pipelines are optional (requirements-fdb.txt)
    groq_plugin = None
    openai_plugin = None
    silero = None


def _nullable_optionals(sig: inspect.Signature) -> inspect.Signature:
    """Declare parameters that default to None as nullable (``float = None`` -> ``float | None``).

    The official tools annotate some optional arguments as plain types with a
    None default (``search_products(max_price: float = None)``), so the schema
    says "a number is required". A model that honestly omits it by sending null
    fails validation and then invents a value (``max_price: 0.0``) the user never
    asked for. Nulls are dropped before the call, so the tool sees no argument.
    """
    params = [
        p.replace(annotation=Optional[p.annotation])
        if p.default is None and p.annotation is not inspect.Parameter.empty
        else p
        for p in sig.parameters.values()
    ]
    return sig.replace(parameters=params)


def _untyped_backend_params(name: str) -> set[str]:
    """Parameters the backend API itself declares as ``Any``.

    The voice tool's schema can only say "string" for them (``value: str``),
    while the API behind it takes any JSON value; a literal "true" or "42"
    there is passed on as the boolean or number it is.
    """
    backend = getattr(registry, "FUNCTIONS", {}).get(name) if registry is not None else None
    if backend is None:
        return set()
    return {p.name for p in inspect.signature(backend).parameters.values() if p.annotation in (Any, "Any")}


def wrap_assistant_tools(
    adapter: FDBBackspaceAdapter, fnc_ctx: AssistantFnc, *, nullable_optionals: bool = False
) -> list[Any]:
    """Connect existing AssistantFnc tool dispatch through FDBBackspaceAdapter.

    ``nullable_optionals`` (used by the Groq pipeline) marks None-defaulted
    parameters nullable in the tool schema; off, the schema is the official one.
    """
    # Suppress immediate file logging from AssistantFnc so only the adapter buffers & flushes
    fnc_ctx.log_tool_call = lambda *args, **kwargs: None

    tools = llm.find_function_tools(fnc_ctx)
    for tool in tools:
        fn_name = tool.info.name
        tool_sig = inspect.signature(tool)
        if nullable_optionals:
            tool_sig = _nullable_optionals(tool_sig)
        orig_func = getattr(tool, "_func", None)

        def _make_wrapper(name: str, sig: inspect.Signature, orig: Any = None):
            async def _adapted_tool(*args, **kwargs):
                call_args = args
                if call_args and (
                    call_args[0] is fnc_ctx
                    or (AssistantFnc is not None and isinstance(call_args[0], AssistantFnc))
                    or (fnc_ctx is not None and isinstance(call_args[0], type(fnc_ctx)))
                ):
                    call_args = call_args[1:]
                # Extract optional invocation meta-parameters if supplied
                work = kwargs.pop("work", None)
                parent_work_id = kwargs.pop("parent_work_id", None)
                turn_id = kwargs.pop("turn_id", "")

                bound = sig.bind_partial(*call_args, **kwargs)
                bound.apply_defaults()
                resolved_kwargs = {k: v for k, v in bound.arguments.items() if v is not None}

                # If session has updated facts for any missing or None parameters in tool signature, use them
                if adapter.session is not None:
                    for param_name in sig.parameters:
                        if (
                            (param_name not in resolved_kwargs or resolved_kwargs[param_name] is None)
                            and adapter.session.has_fact(param_name)
                        ):
                            fact_val = adapter.session.get_fact(param_name)
                            if fact_val is not None:
                                resolved_kwargs[param_name] = fact_val

                # Rules 2 and 3 (dates as spoken, spelled codes joined), enforced
                # by parameter kind for every call; a year is only dropped when
                # the user never said it.
                if os.getenv("FDB_NORMALIZE_ARGS", "1") != "0":
                    heard = (
                        " ".join(t.content for t in adapter.session.turns if t.role == "user")
                        if adapter.session is not None
                        else ""
                    )
                    resolved_kwargs = normalize_arguments(resolved_kwargs, heard, _untyped_backend_params(name))

                def call_backend(**kw):
                    if registry is not None:
                        return registry.call(name, **kw)
                    return {"status": "error", "message": f"Registry unavailable for {name}"}

                result = adapter.execute_tool(
                    func_name=name,
                    args=resolved_kwargs,
                    call_fn=call_backend,
                    work=work,
                    parent_work_id=parent_work_id,
                    turn_id=turn_id,
                )
                return json.dumps(result)

            if orig is not None:
                _adapted_tool.__name__ = getattr(orig, "__name__", name)
                _adapted_tool.__doc__ = getattr(orig, "__doc__", None)
                _adapted_tool.__annotations__ = {
                    k: v for k, v in getattr(orig, "__annotations__", {}).items() if k != "self"
                }
                # The schema is built from these hints: keep them in step with sig
                for pname, param in sig.parameters.items():
                    if param.annotation is not inspect.Parameter.empty:
                        _adapted_tool.__annotations__[pname] = param.annotation
            _adapted_tool.__signature__ = sig
            return _adapted_tool

        tool._func = _make_wrapper(fn_name, tool_sig, orig_func)
        tool.__signature__ = tool_sig
        if nullable_optionals:
            # LiveKit builds the argument schema from the tool's own type hints,
            # which it shares with the official method: give this tool a fresh
            # dict rather than mutating that one (the Gemini path keeps the
            # official schema).
            tool.__annotations__ = {
                **getattr(tool, "__annotations__", {}),
                **{n: p.annotation for n, p in tool_sig.parameters.items() if p.annotation is not inspect.Parameter.empty},
            }
        setattr(fnc_ctx, fn_name, tool)

    return tools


from app.providers.mock import MockProvider
from app.runtime import AgentRuntime
from app.session import Session

# Added to the benchmark's own agent instructions. General rules for disfluent,
# self-correcting speech and multi-step tool use - none of them names a
# benchmark item, and the examples are made up.
ARGUMENT_RULES = (
    "\n\nHOW TO USE THE TOOLS:\n"
    "1. People think out loud: fillers, pauses, false starts and self-corrections "
    "(\"X - actually no, Y\"). Act only on the user's final intent. Ignore values they "
    "abandoned and never call a tool for something they took back.\n"
    "2. Copy argument values the way the user said them. Do not add details they did not "
    "give: no year on a date given as a month and day - pass \"May 4\", not \"2026-05-04\", "
    "even where a tool's example shows another format - and no city or qualifier on a place "
    "they named without one.\n"
    "3. A code spoken character by character (\"Q-7-X-2\", \"Q 7 X 2\", \"B-4\") is one code: "
    "join it without spaces or dashes (\"Q7X2\", \"B4\").\n"
    "4. If the user refers to something without its details (\"my place\", \"my sister's house\"), "
    "pass their words as the value instead of asking a question.\n"
    "5. A request can need several calls. Make every call it needs, in order; when a later "
    "call needs a value an earlier call returned (an ID, an address), use that exact "
    "returned value.\n"
    "6. Never describe results you did not get from a tool in this conversation. If you "
    "have not called the tool yet, call it; do not invent products, prices or listings.\n"
    "7. Never say something is done (booked, updated, added, changed) unless its tool call "
    "succeeded in this conversation. Before you reply, check that every action the user asked "
    "for has had its own tool call; if one has not, make that call first.\n"
    "8. A required argument can never be null. If the user did not give it, use a sensible "
    "typical value rather than leaving it empty.\n"
    "9. Your reply is spoken aloud: plain sentences, no markdown or lists, and keep it short."
)


ACK_FRAME_MS = 20


class InterjectVoiceAgent(Agent if Agent is not None else object):  # type: ignore[misc]
    """The benchmark's VoiceAgent instructions, unchanged, plus ARGUMENT_RULES.

    Optional, off by default (``FDB_ACK=1`` turns it on): when the user's turn
    ends, a short spoken acknowledgement ("Sure, one moment.") is queued ahead of
    the answer, after a silent lead-in (``FDB_ACK_DELAY_S``), while the model
    and the tools work in parallel. It never enters the chat context and claims
    nothing. First measurement (10 recordings) was inconclusive: typical
    perceived latency fell to ~1 s, but twice it played in a mid-sentence pause
    and several replies were lost - at a time when the NVIDIA endpoint itself was
    stalling (a control run without it did worse). It stays off until measured
    on a healthy endpoint. Without a TTS (the text replay) it is skipped.
    """

    def __init__(self) -> None:
        super().__init__(instructions=VoiceAgent().instructions + ARGUMENT_RULES)
        self._ack_frames: list[Any] | None = None

    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        text = os.getenv("FDB_ACK_TEXT", "Sure, one moment.").strip()
        if os.getenv("FDB_ACK", "0") != "1" or not text or not (new_message.text_content or "").strip():
            return
        tts = self.session.tts
        if tts is None:
            return
        if self._ack_frames is None:
            # rendered once per conversation and reused (a local voice takes ~0.1 s)
            self._ack_frames = [ev.frame async for ev in tts.synthesize(text)]
        frames = self._ack_frames
        if not frames:
            return
        delay_s = float(os.getenv("FDB_ACK_DELAY_S", "1.0"))
        rate, channels = frames[0].sample_rate, frames[0].num_channels
        per_frame = rate * ACK_FRAME_MS // 1000
        silence = rtc.AudioFrame(bytes(per_frame * channels * 2), rate, channels, per_frame)

        async def audio():
            for _ in range(int(delay_s * 1000 / ACK_FRAME_MS)):
                yield silence
            for frame in frames:
                yield frame

        self.session.say(text, audio=audio(), add_to_chat_ctx=False, allow_interruptions=True)


# The default, free pipeline (LK_PROVIDER=nvidia): Groq Whisper hears, a
# tool-calling model on NVIDIA's free API catalog thinks, and a local Piper voice
# speaks. Every part fits a free account for a full 100-recording run.
from app.fdb.pipeline_config import (  # noqa: E402
    CASCADED_PROVIDERS,
    GROQ_PROVIDERS,
    NVIDIA_BASE_URL,
    NVIDIA_PROVIDERS,
    fallback_attempt_timeout,
    fallback_enabled,
    groq_model,
    llm_order,
    nvidia_model,
    tts_choice,
)
from app.fdb.pipeline_config import provider as _provider  # noqa: E402


def resolve_session_components() -> dict[str, Any]:
    """AgentSession components for the configured ``LK_PROVIDER``.

    ``nvidia`` (alias ``nvidia_cascaded``): Groq Whisper STT -> a tool-calling
    LLM on NVIDIA's OpenAI-compatible API (``FDB_LLM_MODEL``) -> local Piper
    TTS, with Silero VAD. Needs ``GROQ_API_KEY`` and ``NVIDIA_API_KEY``.

    ``groq`` (alias ``groq_cascaded``): all on Groq - Whisper STT -> LLM ->
    Orpheus TTS (``FDB_TTS=piper`` swaps in the local voice). Needs only
    ``GROQ_API_KEY``, but a free account's daily limits cover only part of a run.

    Anything else: a native realtime speech model, see ``resolve_realtime_model``.
    """
    provider = _provider()
    if provider in CASCADED_PROVIDERS:
        if groq_plugin is None or openai_plugin is None or silero is None:
            raise RuntimeError(
                f"LK_PROVIDER={provider} needs livekit-plugins-groq/-openai/-silero (server/requirements-fdb.txt)"
            )
        # The declared providers are the only ones used: drop any Gemini key a
        # .env supplied, so no background component quietly calls Gemini either.
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(key, None)
        return {
            "stt": groq_plugin.STT(model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"), language="en"),
            "llm": build_llm(),
            "tts": build_tts(),
            "vad": silero.VAD.load(),
        }
    return {"llm": resolve_realtime_model()}


LLM_SERVED_LOG = Path("/tmp/agent_llm_served.log")


def build_llm() -> Any:
    """The tool-calling LLM of the configured cascaded pipeline (also used by the text replay).

    With ``FDB_LLM_FALLBACK=1`` it is a LiveKit ``FallbackAdapter`` over the
    primary and the other provider's model (see ``pipeline_config.fallback_enabled``):
    an attempt that errors, is rate-limited or sends nothing for
    ``FDB_LLM_ATTEMPT_TIMEOUT`` seconds goes to the next model, and a failed
    model is re-checked in the background and used again once it recovers.
    Every answered request is logged with the model that served it.
    """
    nvidia_first = _provider() in NVIDIA_PROVIDERS
    primary = build_nvidia_llm() if nvidia_first else build_groq_llm()
    if not fallback_enabled():
        return primary
    try:
        backup = build_groq_llm() if nvidia_first else build_nvidia_llm()
    except RuntimeError as err:  # the backup's key is missing: run on the primary alone
        logging.warning("LLM fallback unavailable, using the primary only: %s", err)
        return primary
    instances = [(primary, llm_order()[0]), (backup, llm_order()[1])]
    for instance, name in instances:
        instance.on("metrics_collected", functools.partial(_log_llm_served, name))
    adapter = llm.FallbackAdapter(
        [primary, backup],
        attempt_timeout=fallback_attempt_timeout(),
        max_retry_per_llm=0,
        retry_interval=0.5,
    )

    def _on_availability(ev: Any) -> None:
        name = next((n for i, n in instances if i is ev.llm), "?")
        logging.warning("LLM fallback: %s is now %s", name, "available" if ev.available else "unavailable")

    adapter.on("llm_availability_changed", _on_availability)
    return adapter


def _log_llm_served(model: str, metrics: Any) -> None:
    """One line per answered LLM request: which model served it (FDB_LLM_FALLBACK runs)."""
    record = {
        "t": round(getattr(metrics, "timestamp", time.time()), 3),
        "model": model,
        "ttft_s": round(getattr(metrics, "ttft", -1.0), 3),
        "duration_s": round(getattr(metrics, "duration", -1.0), 3),
        "cancelled": bool(getattr(metrics, "cancelled", False)),
    }
    logging.info("LLM served by %s (ttft %.2fs)", model, record["ttft_s"])
    try:
        LLM_SERVED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LLM_SERVED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def build_nvidia_llm() -> Any:
    """A tool-calling model behind an OpenAI-compatible API - NVIDIA's by default.

    ``FDB_LLM_BASE_URL`` / ``FDB_LLM_API_KEY_ENV`` point it at any other
    OpenAI-compatible endpoint (Cerebras, a local vLLM, ...).
    """
    if openai_plugin is None:
        raise RuntimeError("the NVIDIA LLM needs livekit-plugins-openai (server/requirements-fdb.txt)")
    key_env = os.getenv("FDB_LLM_API_KEY_ENV", "NVIDIA_API_KEY")
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"{key_env} is not set - add it to the repository-root .env")
    model = nvidia_model()
    options: dict[str, Any] = {}
    if "gpt-oss" in model:
        # Same effort the Groq pipeline was tuned with (the Groq plugin's own
        # default for gpt-oss): a short think keeps the first word prompt.
        options["reasoning_effort"] = os.getenv("FDB_REASONING_EFFORT", "low")
    return openai_plugin.LLM(
        model=model,
        api_key=api_key,
        base_url=os.getenv("FDB_LLM_BASE_URL", NVIDIA_BASE_URL),
        temperature=float(os.getenv("FDB_LLM_TEMPERATURE", "0")),
        # Pinned so a re-run samples the same way wherever the endpoint honours it.
        extra_body={"seed": int(os.getenv("FDB_LLM_SEED", "7"))},
        # Non-OpenAI endpoints get LiveKit's non-strict tool schemas, as the
        # plugin's own Cerebras/SambaNova/... presets do.
        _strict_tool_schema=os.getenv("FDB_LLM_STRICT_TOOLS", "0") == "1",
        **options,
    )


def build_tts() -> Any:
    """Local Piper by default; ``FDB_TTS=orpheus`` uses Groq's hosted voice."""
    if tts_choice() == "orpheus":
        return groq_plugin.TTS(
            model=os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
            voice=os.getenv("GROQ_TTS_VOICE", "autumn"),
        )
    from app.fdb.local_tts import PiperTTS

    return PiperTTS()


def build_groq_llm() -> Any:
    """The tool-calling LLM of the Groq pipeline (also used by the text replay)."""
    if groq_plugin is None:
        raise RuntimeError("the Groq LLM needs livekit-plugins-groq (server/requirements-fdb.txt)")
    return groq_plugin.LLM(
        # gpt-oss-120b: of the Groq models tried, the one that reliably calls the
        # tool instead of describing made-up results or asking for confirmation.
        model=groq_model(),
        temperature=float(os.getenv("GROQ_LLM_TEMPERATURE", "0")),
    )


def resolve_realtime_model() -> Any:
    """Resolve realtime model provider with support for gemini3_8."""
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    provider = _provider()
    if provider in {"gemini3_8", "gemini_3_8", "gemini3.8"}:
        return google.realtime.RealtimeModel(
            model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live"),
            voice=os.getenv("GOOGLE_VOICE", "Puck"),
        )
    if get_realtime_model is not None:
        return get_realtime_model()
    raise RuntimeError("Realtime model provider unavailable")


@dataclass
class FDBRunnerContext:
    room_name: str
    session_state: Session
    core: BackspaceCore
    runtime: AgentRuntime
    adapter: FDBBackspaceAdapter
    tracker: Any
    fnc_ctx: Any
    tools: list[Any]
    tool_map: dict[str, Any]
    session: Any = None
    emitted_frames: list[Any] = field(default_factory=list)

    def get_tool(self, name: str) -> Any:
        return self.tool_map[name]


def create_fdb_runner_context(
    room_name: str,
    *,
    model: Any = None,
    components: dict[str, Any] | None = None,
    nullable_optionals: bool | None = None,
    telemetry_path: str = "/tmp/agent_tool_calls.log",
    emit: Callable[[Any], Awaitable[None]] | None = None,
) -> FDBRunnerContext:
    """Create a fully-wired FDB Backspace runner context connecting LiveKit,
    BackspaceCore, Session, AgentRuntime, and FDBBackspaceAdapter."""
    session_state = Session(session_id=room_name)
    core = session_state.backspace

    emitted_frames: list[Any] = []

    async def _default_emit(frame: Any) -> None:
        emitted_frames.append(frame)

    actual_emit = emit if emit is not None else _default_emit
    runtime = AgentRuntime(session_state, emit=actual_emit)
    # The runtime here only tracks facts for BACKSPACE; its generated text goes
    # to emitted_frames and is never spoken (the voice model answers). Keep it
    # on the offline engine so it never spends a hosted model's quota.
    runtime.provider = MockProvider()
    adapter = FDBBackspaceAdapter(
        core=core,
        room_name=room_name,
        session=session_state,
        telemetry_path=telemetry_path,
    )
    tracker = LatencyTracker() if LatencyTracker is not None else None
    fnc_ctx = AssistantFnc(tracker, room_name) if AssistantFnc is not None else None
    if nullable_optionals is None:
        nullable_optionals = os.getenv("LK_PROVIDER", "").strip().lower() in CASCADED_PROVIDERS
    tools = (
        wrap_assistant_tools(adapter, fnc_ctx, nullable_optionals=nullable_optionals)
        if fnc_ctx is not None
        else []
    )
    tool_map = {t.info.name: t for t in tools}

    session = None
    if components is None and model is not None:
        components = {"llm": model}
    if components and AgentSession is not None:
        # Hard scenarios chain three dependent calls and still need a final
        # spoken answer; LiveKit's default of 3 tool steps can cut that short.
        max_steps = int(os.getenv("FDB_MAX_TOOL_STEPS", "5"))
        # Hosted per-minute token limits answer 429 with a short Retry-After;
        # give the LLM a few more spaced retries than LiveKit's default of 3
        # before a turn is given up. (The plugin's HTTP client does not retry
        # on its own - LiveKit owns the retry policy.)
        conn = SessionConnectOptions(
            llm_conn_options=APIConnectOptions(
                max_retry=int(os.getenv("FDB_LLM_MAX_RETRY", "5")),
                retry_interval=2.0,
                timeout=float(os.getenv("FDB_LLM_TIMEOUT", "20")),
            )
        )
        session = AgentSession(**components, tools=tools, max_tool_steps=max_steps, conn_options=conn)

    return FDBRunnerContext(
        room_name=room_name,
        session_state=session_state,
        core=core,
        runtime=runtime,
        adapter=adapter,
        tracker=tracker,
        fnc_ctx=fnc_ctx,
        tools=tools,
        tool_map=tool_map,
        session=session,
        emitted_frames=emitted_frames,
    )


server = AgentServer() if AgentServer is not None else None


if server is not None:

    @server.rtc_session()
    async def entrypoint(ctx: agents.JobContext) -> None:
        runner_ctx = create_fdb_runner_context(ctx.room.name, components=resolve_session_components())
        adapter = runner_ctx.adapter
        tracker = runner_ctx.tracker
        runtime = runner_ctx.runtime
        session = runner_ctx.session

        session_done = asyncio.Event()

        # 2. Attach shutdown hook using official LiveKit JobContext API
        async def _on_shutdown():
            adapter.flush()
            session_done.set()

        ctx.add_shutdown_callback(_on_shutdown)

        if session is not None:

            @session.on("close")
            def _on_session_close(*args):
                adapter.flush()
                session_done.set()

            @session.on("user_input_transcribed")
            def on_user_input(msg: Any):
                if tracker and not tracker.query_received:
                    tracker.user_done_at = time.time()
                    tracker.query_received = True
                if getattr(msg, "is_final", False) and getattr(msg, "transcript", "").strip():
                    asyncio.create_task(runtime.on_final(msg.transcript))

            @session.on("agent_state_changed")
            def on_agent_state(ev: Any):
                if getattr(ev, "new_state", "") == "speaking" and tracker and tracker.query_received and not tracker.agent_start_at:
                    tracker.agent_start_at = time.time()
                    tracker.log_breakdown(tool_name="Adapted Tool", room_name=ctx.room.name)
                    tracker.reset()

        @ctx.room.on("disconnected")
        def _on_room_disconnected(*args):
            adapter.flush()
            session_done.set()

        try:
            if session is not None:
                await session.start(room=ctx.room, agent=InterjectVoiceAgent(), record=False)
            await session_done.wait()
        finally:
            adapter.flush()


if __name__ == "__main__":
    if agents is not None and server is not None:
        agents.cli.run_app(server)
