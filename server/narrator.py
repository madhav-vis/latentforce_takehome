"""Step 4: progress events -> screen (always) + voice (only in awkward silences).

Every tool call becomes an event from a template (no model call). The screen gets the
exact detail ("Reading src/auth/token_service.py, lines 40-100"); the voice gets a short
spoken form ("Reading the token service") and only when the Scheduler finds a gap.
"""

import asyncio
import re
import time
from pathlib import PurePosixPath
from typing import Awaitable, Callable

GENERIC_STEMS = {"index", "__init__", "mod", "main", "init"}
# Names that already end in a noun read fine alone ("the token service"); others get
# "file" ("the use auth file", "the app file").
ROLE_WORDS = {"service", "services", "controller", "handler", "handlers", "router", "routes",
              "model", "models", "manager", "client", "provider", "store", "module", "helpers",
              "utils", "config", "settings", "schema", "schemas", "views", "middleware", "hook",
              "component", "page", "test", "tests", "spec", "types", "api", "cli"}


def words(name: str) -> str:
    """token_service / tokenService / token-service -> 'token service'."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[_\-.\s]+", " ", name).strip().lower()


def short(path: str, *, folder: bool = False) -> str:
    """Speakable name for a path: src/services/auth/token_service.py -> 'the token service'."""
    p = PurePosixPath(path.strip().strip("/") or ".")
    if str(p) in (".", ""):
        return "the top level of the repo"
    if folder:
        return f"the {words(p.name)} folder"
    stem = p.stem if p.suffix else p.name
    if stem in GENERIC_STEMS and len(p.parts) > 1:
        return f"the {words(p.parent.name)} module"
    w = words(stem) or stem
    return f"the {w}" + ("" if w.split()[-1] in ROLE_WORDS else " file")


def _speakable_pattern(pattern: str) -> str | None:
    """Plain identifiers read fine aloud; regexes don't."""
    p = pattern.strip()
    if re.fullmatch(r"(?:def |class |function )?[A-Za-z_][\w ]{0,40}", p) and len(p.split()) <= 3:
        return words(p.split()[-1])
    return None


def _lines(i: dict) -> str:
    a, b = i.get("start"), i.get("end")
    if a and b:
        return f", lines {a}-{b}"
    return f", from line {a}" if a else ""


def describe(name: str, i: dict) -> tuple[str, str | None]:
    """(screen text, spoken text or None) for one tool call."""
    i = i if isinstance(i, dict) else {}
    if name == "grep":
        pat, glob = i.get("pattern", ""), i.get("path_glob")
        spoken = _speakable_pattern(pat)
        return (f"Searching for “{pat}”" + (f" in {glob}" if glob else ""),
                f"Searching for {spoken}" if spoken else "Searching the code")
    if name == "read_file":
        path = i.get("path", "")
        return f"Reading {path}{_lines(i)}", f"Reading {short(path)}"
    if name == "list_dir":
        path = i.get("path", ".")
        return f"Looking in {path}", f"Looking in {short(path, folder=True)}"
    if name == "show_code":
        return f"Showing {i.get('path', '')}{_lines(i)}", None     # the screen speaks for itself
    if name == "deep_dive":
        return "Tracing it across files", "Tracing it across files"
    return "Working…", None


def tool_event(name: str, inp: dict) -> dict:
    screen, spoken = describe(name, inp)
    return {"type": "progress", "text": screen, "say": spoken, "tool": name, "ts": time.monotonic()}


class Scheduler:
    """Owns "who may speak now" for progress lines.

    Priority: user speech > answer audio > progress lines. A progress line is spoken only if
    the user isn't talking, no answer audio is playing or about to play, nothing has been
    heard for GAP_S, and the event is under FRESH_S old. Only the newest event is kept;
    stale ones are dropped, never queued.
    """

    GAP_S = 2.5
    FRESH_S = 3.0
    TICK_S = 0.25
    PENDING_TIMEOUT_S = 3.0     # text sent to TTS that never started playing

    def __init__(self, speak: Callable[[str], Awaitable[None]]):
        self.speak = speak
        self.user_talking = False
        self.bot_talking = False
        self.last_audio = time.monotonic()
        self.pending: dict | None = None
        self._speech_sent_at: float | None = None   # answer text handed to TTS, not yet audible
        self._busy_until = 0.0                      # raw audio we pushed ourselves is playing

    # -- inputs --
    def on_progress(self, ev: dict):
        if ev.get("say"):
            self.pending = ev                      # keep only the newest

    def speech_queued(self):
        self._speech_sent_at = time.monotonic()

    def playing_for(self, seconds: float):
        """Pre-rendered audio pushed as raw frames doesn't produce bot-speaking frames, so
        the caller says how long it lasts."""
        self._busy_until = time.monotonic() + seconds
        self.last_audio = max(self.last_audio, self._busy_until)

    def on_bot_started(self):
        self.bot_talking = True
        self._speech_sent_at = None

    def on_bot_stopped(self):
        self.bot_talking = False
        self.last_audio = time.monotonic()

    def on_user_started(self):
        self.user_talking = True
        self.pending = None

    def on_user_stopped(self):
        self.user_talking = False
        self.last_audio = time.monotonic()        # the question counts as the last audio

    def clear(self):
        self.pending = None
        self._speech_sent_at = None
        self._busy_until = 0.0

    # -- decisions --
    def quiet_for(self, now: float | None = None) -> float:
        return max(0.0, (now or time.monotonic()) - self.last_audio)

    def speech_on_the_way(self, now: float) -> bool:
        return self._speech_sent_at is not None and now - self._speech_sent_at < self.PENDING_TIMEOUT_S

    def can_speak(self, now: float | None = None) -> bool:
        now = now or time.monotonic()
        return not (self.user_talking or self.bot_talking or now < self._busy_until
                    or self.speech_on_the_way(now))

    async def tick(self):
        ev, now = self.pending, time.monotonic()
        if not ev or not self.can_speak(now):
            return
        if now - ev["ts"] > self.FRESH_S:
            self.pending = None                    # stale: drop it
            return
        if self.quiet_for(now) > self.GAP_S:
            self.pending = None
            self.last_audio = now
            await self.speak(ev["say"])

    async def run(self):
        while True:
            await asyncio.sleep(self.TICK_S)
            await self.tick()
