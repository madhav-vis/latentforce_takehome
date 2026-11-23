"""FastAPI server: /session, /offer (WebRTC), /events (SSE), and the page.

    .venv/bin/python -m server.main            # http://localhost:8000

Setup starts the moment the link is posted (POST /session), so the clone and map
build while the browser asks for the microphone.
"""

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pydantic import BaseModel

from .pipeline import prerender_fallback_greeting, run_bot
from .session import SESSIONS, Session, emit
from .setup import SetupError, parse_repo_url, setup_repo

WEB = Path(__file__).resolve().parent.parent / "web"
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "3"))
CONNECT_WITHIN_S = 300       # drop sessions whose browser never connects

webrtc = SmallWebRTCRequestHandler()
_tasks: set[asyncio.Task] = set()
_connected: set[str] = set()


def _spawn(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)
    return t


def _drop_session(s: Session):
    SESSIONS.pop(s.id, None)
    _connected.discard(s.id)
    if s.repo_path:
        shutil.rmtree(s.repo_path, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await prerender_fallback_greeting()
    yield
    await webrtc.close()
    for s in list(SESSIONS.values()):
        _drop_session(s)


app = FastAPI(lifespan=lifespan)


class SessionRequest(BaseModel):
    repo_url: str


async def _setup(s: Session):
    try:
        await setup_repo(s)
    except SetupError as e:
        s.error = str(e)
        s.ready.set()                       # unblock the greeting so it can report the error
    except Exception as e:
        logger.exception("setup failed")
        s.error = "Setup failed."
        await emit(s, "error", text=s.error)
        s.ready.set()


@app.post("/session")
async def create_session(body: SessionRequest):
    if not parse_repo_url(body.repo_url):
        raise HTTPException(400, "Expected a link like https://github.com/owner/repo")
    if len(SESSIONS) >= MAX_SESSIONS:
        raise HTTPException(429, "Too many sessions right now; try again in a minute.")
    s = Session(repo_url=body.repo_url.strip())
    SESSIONS[s.id] = s
    _spawn(_setup(s))
    _spawn(_expire_if_unconnected(s))
    return {"session_id": s.id}


async def _expire_if_unconnected(s: Session):
    await asyncio.sleep(CONNECT_WITHIN_S)
    if s.id in SESSIONS and s.id not in _connected:
        logger.info(f"session {s.id}: never connected, dropping")
        _drop_session(s)


@app.get("/events/{session_id}")
async def events(session_id: str, request: Request):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "unknown session")

    async def stream():
        while True:
            if await request.is_disconnected():
                return
            try:
                ev = await asyncio.wait_for(s.events.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/offer")
async def offer(request: SmallWebRTCRequest, session_id: str, background_tasks: BackgroundTasks):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "unknown session")

    async def on_connection(connection: SmallWebRTCConnection):
        _connected.add(s.id)

        async def run():
            try:
                await run_bot(connection, s)
            finally:
                _drop_session(s)
        background_tasks.add_task(run)

    return await webrtc.handle_web_request(request=request, webrtc_connection_callback=on_connection)


@app.patch("/offer")
async def ice_candidates(request: SmallWebRTCPatchRequest, session_id: str):
    await webrtc.handle_patch_request(request)
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
