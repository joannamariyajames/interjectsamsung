"""Standalone LiveKit Realtime Voice Agent worker for Gemini 3.8.

Bridges LiveKit WebRTC room audio (mic input & speaker output) to Google RealtimeModel.
Runs as a standalone LiveKit worker process using cli.run_app(WorkerOptions(...)).
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins.google.realtime import RealtimeModel

from .config import settings

logger = logging.getLogger("livekit_worker")


def build_realtime_model(
    api_key: str | None = None,
    model: str | None = None,
    instructions: str = "You are an interruptible real-time voice assistant.",
) -> RealtimeModel:
    """Instantiate Google RealtimeModel for LiveKit audio streaming."""
    key = api_key or settings.gemini_api_key
    if not key or not key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured for LiveKit RealtimeModel.")

    model_name = model or settings.gemini_live_model
    return RealtimeModel(
        model=model_name,
        api_key=key,
        instructions=instructions,
    )


def create_agent_session(
    model: RealtimeModel | None = None,
    instructions: str = "You are an interruptible real-time voice assistant.",
) -> tuple[Agent, AgentSession]:
    """Construct Agent and AgentSession with the configured RealtimeModel."""
    realtime_model = model or build_realtime_model(instructions=instructions)
    agent = Agent(instructions=instructions)
    session = AgentSession(llm=realtime_model)
    return agent, session


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit agent worker job entrypoint."""
    logger.info("Connecting to LiveKit room %s...", ctx.room.name)
    await ctx.connect()
    logger.info("Connected to room %s. Initializing agent session...", ctx.room.name)

    agent, session = create_agent_session()
    await session.start(agent, room=ctx.room)
    logger.info("Agent session active in room %s.", ctx.room.name)


def main() -> None:
    """Launch the LiveKit worker process using configuration settings."""
    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=settings.livekit_url or "",
        api_key=settings.livekit_api_key or "",
        api_secret=settings.livekit_api_secret or "",
    )
    cli.run_app(opts)


if __name__ == "__main__":
    main()
