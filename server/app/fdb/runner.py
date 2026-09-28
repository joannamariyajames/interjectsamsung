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
from typing import Any, Awaitable, Callable

# Resolve Full-Duplex-Bench/v3 path as read-only external dependency
_REPO_ROOT = Path(__file__).resolve().parents[3]  # interjectsamsung
_FDB_V3_DIR = _REPO_ROOT.parent / "Full-Duplex-Bench" / "v3"
if _FDB_V3_DIR.exists() and str(_FDB_V3_DIR) not in sys.path:
    sys.path.insert(0, str(_FDB_V3_DIR))

# Ensure interjectsamsung/server is in sys.path for app.backspace and app.fdb
_SERVER_DIR = _REPO_ROOT / "server"
if _SERVER_DIR.exists() and str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

try:
    from dotenv import load_dotenv

    _ENV_LOCAL = _REPO_ROOT / ".env.local"
    if _ENV_LOCAL.exists():
        load_dotenv(_ENV_LOCAL)
except ImportError:
    pass

from app.backspace import BackspaceCore
from app.fdb.adapter import FDBBackspaceAdapter

try:
    from livekit import agents
    from livekit.agents import AgentServer, AgentSession, llm
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
    AgentServer = None
    AgentSession = None
    llm = None
    google = None
    AssistantFnc = None
    VoiceAgent = None
    get_realtime_model = None
    LatencyTracker = None
    registry = None


def wrap_assistant_tools(adapter: FDBBackspaceAdapter, fnc_ctx: AssistantFnc) -> list[Any]:
    """Connect existing AssistantFnc tool dispatch through FDBBackspaceAdapter."""
    # Suppress immediate file logging from AssistantFnc so only the adapter buffers & flushes
    fnc_ctx.log_tool_call = lambda *args, **kwargs: None

    tools = llm.find_function_tools(fnc_ctx)
    for tool in tools:
        fn_name = tool.info.name
        tool_sig = inspect.signature(tool)
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
            _adapted_tool.__signature__ = sig
            return _adapted_tool

        tool._func = _make_wrapper(fn_name, tool_sig, orig_func)
        tool.__signature__ = tool_sig
        setattr(fnc_ctx, fn_name, tool)

    return tools


from app.runtime import AgentRuntime
from app.session import Session


def resolve_realtime_model() -> Any:
    """Resolve realtime model provider with support for gemini3_8."""
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    provider = os.getenv("LK_PROVIDER", "gemini3_8").strip().lower()
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
    adapter = FDBBackspaceAdapter(
        core=core,
        room_name=room_name,
        session=session_state,
        telemetry_path=telemetry_path,
    )
    tracker = LatencyTracker() if LatencyTracker is not None else None
    fnc_ctx = AssistantFnc(tracker, room_name) if AssistantFnc is not None else None
    tools = wrap_assistant_tools(adapter, fnc_ctx) if fnc_ctx is not None else []
    tool_map = {t.info.name: t for t in tools}

    session = None
    if model is not None and AgentSession is not None:
        session = AgentSession(llm=model, tools=tools)

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
        model = resolve_realtime_model()
        runner_ctx = create_fdb_runner_context(ctx.room.name, model=model)
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
                await session.start(room=ctx.room, agent=VoiceAgent(), record=False)
            await session_done.wait()
        finally:
            adapter.flush()


if __name__ == "__main__":
    if agents is not None and server is not None:
        agents.cli.run_app(server)
