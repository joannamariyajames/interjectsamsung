"""FastAPI surface: one full-duplex websocket plus a little metadata."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import COOKIE_NAME, SESSION_TTL_S, AuthError, AuthStore, User
from .config import settings
from .drive.runtime import DriveRuntime
from .providers import build_provider
from .retrieval import corpus
from .runtime import AgentRuntime
from .schemas import ErrorFrame, ServerFrame, Stage, StageFrame
from .session import store
from .tools import REGISTRY

app = FastAPI(
    title="Interject",
    description="An interruptible, full-duplex real-time agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) + ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- accounts -------------------------------------------------------------------
_auth: AuthStore | None = None


def auth_store() -> AuthStore:
    """The account store, opened on first use (``AUTH_DB_PATH``)."""
    global _auth
    if _auth is None:
        _auth = AuthStore()
    return _auth


class SignupBody(BaseModel):
    name: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


def _set_session_cookie(request: Request, response: Response, user: User) -> None:
    response.set_cookie(
        COOKIE_NAME, auth_store().start_session(user), max_age=SESSION_TTL_S, httponly=True,
        samesite="lax", secure=request.url.scheme == "https", path="/",
    )


def _auth_error(err: AuthError) -> HTTPException:
    status = {"email_taken": 409, "invalid_credentials": 401, "too_many_attempts": 429}.get(err.code, 400)
    return HTTPException(status_code=status, detail={"code": err.code, "message": str(err)})


def current_user(request: Request) -> User | None:
    return auth_store().user_for(request.cookies.get(COOKIE_NAME))


def require_user(request: Request) -> User | None:
    """Gate for the app's endpoints; ``AUTH_REQUIRED=0`` opens them (local development)."""
    user = current_user(request)
    if user is None and settings.auth_required:
        raise HTTPException(status_code=401, detail={"code": "auth_required", "message": "Please log in."})
    return user


@app.post("/api/auth/signup", status_code=201)
async def signup(body: SignupBody, request: Request, response: Response) -> dict[str, Any]:
    try:
        user = auth_store().create_user(body.name, body.email, body.password)
    except AuthError as err:
        raise _auth_error(err) from None
    _set_session_cookie(request, response, user)
    return {"user": user.to_dict()}


@app.post("/api/auth/login")
async def login(body: LoginBody, request: Request, response: Response) -> dict[str, Any]:
    try:
        user = auth_store().authenticate(body.email, body.password)
    except AuthError as err:
        raise _auth_error(err) from None
    _set_session_cookie(request, response, user)
    return {"user": user.to_dict()}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, Any]:
    auth_store().end_session(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
async def me(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail={"code": "auth_required", "message": "Please log in."})
    return {"user": user.to_dict()}


async def _reject_unauthenticated(socket: WebSocket) -> bool:
    """Accept, explain and close (code 4401) a socket without a valid session."""
    if not settings.auth_required or auth_store().user_for(socket.cookies.get(COOKIE_NAME)):
        return False
    await socket.accept()
    await socket.send_text(json.dumps({"t": "error", "code": "auth_required", "message": "Please log in."}))
    await socket.close(code=4401)
    return True


@app.get("/api/health")
async def health() -> dict[str, Any]:
    # Same precedence as providers.build_provider().
    if settings.use_gemini:
        provider, model = "gemini", settings.gemini_model
    elif settings.use_real_llm:
        provider, model = "openai-compatible", settings.llm_model
    else:
        provider, model = "local-deterministic", "deterministic"
    backup = settings.llm_fallback_model if provider == "openai-compatible" and settings.llm_fallback_api_key else None
    return {
        "status": "ok",
        "provider": provider,
        "model": model,
        "fallback_model": backup,
        "corpus_documents": len(corpus.docs),
        "active_sessions": len(store),
        "speculation": settings.speculation_enabled,
        "strict_harness": settings.strict_harness,
    }


@app.get("/api/corpus", dependencies=[Depends(require_user)])
async def list_corpus() -> dict[str, Any]:
    return {
        "documents": [
            {"doc_id": d.doc_id, "source": d.source, "title": d.title, "snippet": d.snippet}
            for d in corpus.docs
        ]
    }


@app.get("/api/tools", dependencies=[Depends(require_user)])
async def list_tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "effect": spec.effect.value,
                "params": list(spec.params),
                "needs_confirmation": spec.confirm,
            }
            for spec in REGISTRY.values()
        ]
    }


@app.get("/api/transcripts", dependencies=[Depends(require_user)])
async def transcripts() -> dict[str, Any]:
    """Canned speech transcripts for the simulated voice input.

    The brief allows voice to be simulated from transcripts. The client replays
    these word by word at a speaking cadence, emitting the same partial frames a
    real recogniser would, so the full-duplex path is exercised honestly rather
    than by pasting a finished sentence.
    """
    return {
        "transcripts": [
            {
                "id": "baggage",
                "label": "Baggage question",
                "text": "what are the baggage limits on each cabin",
                "wpm": 150,
            },
            {
                "id": "flight",
                "label": "Flight request",
                "text": "find me a flight from Bengaluru to Mumbai on Friday morning",
                "wpm": 165,
            },
            {
                "id": "refund",
                "label": "Mid-thought swerve",
                "text": "actually wait what happens to my refund if I cancel this",
                "wpm": 180,
            },
            {
                "id": "delay",
                "label": "Disruption",
                "text": "my flight is delayed overnight what am I entitled to",
                "wpm": 155,
            },
        ]
    }


@app.get("/api/scenarios", dependencies=[Depends(require_user)])
async def scenarios() -> dict[str, Any]:
    """Scripted demos. The UI can replay these to show each behaviour on cue."""
    return {
        "scenarios": [
            {
                "id": "barge-in",
                "title": "Barge-in mid-answer",
                "blurb": "Interrupt while the agent is streaming, then continue. It picks up from the checkpoint instead of restarting.",
                "steps": [
                    {"kind": "say", "text": "What are the baggage limits on each cabin?"},
                    {"kind": "wait_tokens", "count": 14},
                    {"kind": "interrupt"},
                    {"kind": "say", "text": "go on"},
                ],
            },
            {
                "id": "goal-switch",
                "title": "Goal switch and return",
                "blurb": "Swerve to a different goal mid-task, then come back. The first goal is parked, not lost.",
                "steps": [
                    {"kind": "say", "text": "Find me a flight from Bengaluru to Mumbai on Friday"},
                    {"kind": "wait_idle"},
                    {"kind": "say", "text": "actually, what happens to my refund if I cancel?"},
                    {"kind": "wait_idle"},
                    {"kind": "say", "text": "anyway, back to the flight"},
                ],
            },
            {
                "id": "refine",
                "title": "Constraint accumulates",
                "blurb": "A terse amendment with no shared nouns still lands on the right goal.",
                "steps": [
                    {"kind": "say", "text": "Which hotel should I book in Mumbai?"},
                    {"kind": "wait_idle"},
                    {"kind": "say", "text": "make it under 9000 a night"},
                ],
            },
            {
                "id": "harness",
                "title": "Harness refuses an irreversible call",
                "blurb": "The agent cannot self-authorise a booking. The refusal is shown, not hidden.",
                "steps": [{"kind": "say", "text": "just book the ticket for me now"}],
            },
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    if await _reject_unauthenticated(socket):
        return
    await socket.accept()
    session = store.get_or_create(None)
    send_lock = asyncio.Lock()
    closed = False

    async def emit(frame: ServerFrame) -> None:
        nonlocal closed
        if closed:
            return
        try:
            async with send_lock:
                await socket.send_text(frame.model_dump_json())
        except (WebSocketDisconnect, RuntimeError):
            closed = True

    runtime = AgentRuntime(session, emit)

    await socket.send_text(
        json.dumps(
            {
                "t": "ready",
                "session_id": session.session_id,
                "provider": runtime.provider.name,
                "speculation": settings.speculation_enabled,
                "token_delay_ms": settings.token_delay_ms,
            }
        )
    )
    await emit(StageFrame(stage=Stage.IDLE, detail="Connected. Start talking."))

    try:
        while True:
            raw = await socket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await emit(ErrorFrame(message="Malformed frame ignored."))
                continue

            kind = event.get("t")
            # Note the ordering: interrupt is handled before anything that
            # could await, so a barge-in is never queued behind a turn.
            if kind == "interrupt":
                await runtime.interrupt(event.get("reason", "barge_in"))
            elif kind == "partial":
                await runtime.on_partial(str(event.get("text", "")))
            elif kind == "final":
                await runtime.on_final(
                    str(event.get("text", "")),
                    modality=event.get("modality", "text"),
                    attachment=event.get("attachment"),
                )
            elif kind == "resume":
                await runtime.resume()
            elif kind == "reset":
                await runtime.reset()
            elif kind == "config":
                runtime.configure(
                    token_delay_ms=event.get("token_delay_ms"),
                    strict_harness=event.get("strict_harness"),
                )
                await emit(StageFrame(stage=Stage.IDLE, detail="Settings applied."))
            elif kind == "ping":
                await emit(StageFrame(stage=Stage.IDLE, detail="pong"))
            else:
                await emit(ErrorFrame(message=f"Unknown frame {kind!r}."))
    except WebSocketDisconnect:
        pass
    finally:
        closed = True
        await runtime.shutdown()
        # Session-scoped memory: the socket closing is the end of it.
        store.drop(session.session_id)


@app.websocket("/ws/drive")
async def drive_socket(socket: WebSocket) -> None:
    """The in-car voice assistant (use-case extension): same framing as /ws."""
    if await _reject_unauthenticated(socket):
        return
    await socket.accept()
    send_lock = asyncio.Lock()
    closed = False

    async def emit(frame: ServerFrame) -> None:
        nonlocal closed
        if closed:
            return
        try:
            async with send_lock:
                await socket.send_text(frame.model_dump_json())
        except (WebSocketDisconnect, RuntimeError):
            closed = True

    provider = build_provider()  # only consulted for questions outside driving
    runtime = DriveRuntime(emit, provider=provider)
    await socket.send_text(json.dumps({"t": "ready", "mode": "drive", "provider": provider.name,
                                       "origin": runtime.origin.label}))
    await runtime.start()
    try:
        while True:
            raw = await socket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await emit(ErrorFrame(message="Malformed frame ignored."))
                continue
            kind = event.get("t")
            if kind == "interrupt":  # handled first: a barge-in never waits behind a turn
                heard = event.get("heard_chars")
                await runtime.interrupt(int(heard) if isinstance(heard, (int, float)) else None)
            elif kind == "partial":
                await runtime.on_partial(str(event.get("text", "")))
            elif kind == "final":
                await runtime.on_final(str(event.get("text", "")), modality=str(event.get("modality", "voice")))
            elif kind == "resume":
                await runtime.on_final("go on", modality="text")
            elif kind == "ping":
                await emit(StageFrame(stage=Stage.IDLE, detail="pong"))
            else:
                await emit(ErrorFrame(message=f"Unknown frame {kind!r}."))
    except WebSocketDisconnect:
        pass
    finally:
        closed = True
        await runtime.shutdown()
