"""Member 4 FDB-v3 LiveKit Agent Runner with Backspace Adapter.

Reuses official Full-Duplex-Bench components as a read-only dependency.
Does NOT modify or copy official FDB files.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

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

        def _make_wrapper(name: str, sig: inspect.Signature):
            async def _adapted_tool(*args, **kwargs):
                call_args = args
                if call_args and call_args[0] is fnc_ctx:
                    call_args = call_args[1:]
                bound = sig.bind_partial(*call_args, **kwargs)
                bound.apply_defaults()
                resolved_kwargs = {k: v for k, v in bound.arguments.items() if v is not None}

                def call_backend(**kw):
                    if registry is not None:
                        return registry.call(name, **kw)
                    return {"status": "error", "message": f"Registry unavailable for {name}"}

                result = adapter.execute_tool(
                    func_name=name,
                    args=resolved_kwargs,
                    call_fn=call_backend,
                )
                return json.dumps(result)

            return _adapted_tool

        tool._func = _make_wrapper(fn_name, tool_sig)

    return tools


from app.runtime import AgentRuntime
from app.session import Session


def resolve_realtime_model() -> Any:
    """Resolve realtime model provider with support for gemini3_8."""
    provider = os.getenv("LK_PROVIDER", "gpt_realtime").lower()
    if provider in {"gemini3_8", "gemini_3_8", "gemini3.8"}:
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        return google.realtime.RealtimeModel(
            model="gemini-3.8-live",
            voice=os.getenv("GOOGLE_VOICE", "Puck"),
        )
    if get_realtime_model is not None:
        return get_realtime_model()
    raise RuntimeError("Realtime model provider unavailable")


server = AgentServer() if AgentServer is not None else None


if server is not None:

    @server.rtc_session()
    async def entrypoint(ctx: agents.JobContext) -> None:
        model = resolve_realtime_model()
        tracker = LatencyTracker()

        # 1. Create room-scoped Session, BackspaceCore, AgentRuntime and Adapter
        session_state = Session(session_id=ctx.room.name)
        core = session_state.backspace
        runtime = AgentRuntime(session_state, emit=lambda frame: asyncio.sleep(0))
        adapter = FDBBackspaceAdapter(core=core, room_name=ctx.room.name)

        session_done = asyncio.Event()

        # 2. Attach shutdown hook using official LiveKit JobContext API
        async def _on_shutdown():
            adapter.flush()
            session_done.set()

        ctx.add_shutdown_callback(_on_shutdown)

        # 3. Create official AssistantFnc and adapt tool dispatch
        fnc_ctx = AssistantFnc(tracker, ctx.room.name)
        tools = wrap_assistant_tools(adapter, fnc_ctx)

        session = AgentSession(llm=model, tools=tools)

        @session.on("close")
        def _on_session_close(*args):
            adapter.flush()
            session_done.set()

        @ctx.room.on("disconnected")
        def _on_room_disconnected(*args):
            adapter.flush()
            session_done.set()

        @session.on("user_input_transcribed")
        def on_user_input(msg: Any):
            if not tracker.query_received:
                tracker.user_done_at = time.time()
                tracker.query_received = True
            if getattr(msg, "is_final", False) and getattr(msg, "transcript", "").strip():
                asyncio.create_task(runtime.on_final(msg.transcript))

        @session.on("agent_state_changed")
        def on_agent_state(ev: Any):
            if getattr(ev, "new_state", "") == "speaking" and tracker.query_received and not tracker.agent_start_at:
                tracker.agent_start_at = time.time()
                tracker.log_breakdown(tool_name="Adapted Tool", room_name=ctx.room.name)
                tracker.reset()

        try:
            await session.start(room=ctx.room, agent=VoiceAgent())
            await session_done.wait()
        finally:
            adapter.flush()


if __name__ == "__main__":
    if agents is not None and server is not None:
        agents.cli.run_app(server)
