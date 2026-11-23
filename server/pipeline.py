"""Steps 2 and 5a: the voice pipeline.

    mic (WebRTC) -> Deepgram Flux -> user turns -> RepoAgent -> ElevenLabs -> speaker (WebRTC)

Flux does the end-of-turn detection and proposes turn starts/stops; UserTurnProcessor
turns those into UserStarted/StoppedSpeaking frames and interruptions (barge-in).
RepoAgent runs the step-3 answer() loop on each final transcript and streams cleaned
sentences to the TTS. A Scheduler (step 4) speaks progress lines only in silent gaps, and
HeardTracker, after the output transport, records which words were actually played so a
barge-in can tell the model what the user heard (step 5d).

Pipecat 1.16 names (checked against the installed source): PipelineWorker (was
PipelineTask), WorkerRunner (was PipelineRunner), and interruptions come from user
turn strategies rather than an allow_interruptions flag.
"""

import asyncio
import copy
import dataclasses
import os
import re
import time

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EagerTranscriptionFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from .agent import answer
from .narrator import Scheduler, short, tool_event
from .session import Session, emit

TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"      # ElevenLabs premade "Rachel"
OUT_SAMPLE_RATE = 24000
GREETING_WAIT_S = 1.5
# Step 5e: start answering on Flux's predicted end of turn and release the audio when the
# turn is committed. EAGER_EOT=0 turns it off. Voice harness, 12 questions each: at Flux's
# default 0.5 the prediction landed only 0.01-0.12 s before the commit (p50 1.68 s to first
# audio); at 0.3 it lands up to 0.64 s earlier (p50 1.44 s, p95 1.64 s, from 1.82 / 2.06 s
# with eager off). Withdrawn predictions cost one discarded model call each.
EAGER_END_OF_TURN = os.environ.get("EAGER_EOT", "1") != "0"
EAGER_EOT_THRESHOLD = float(os.environ.get("EAGER_EOT_THRESHOLD", "0.3"))
CLONE_WAIT_S = 30

FALLBACK_GREETING = "Hi! I'm reading through the repo now; ask me anything and I'll start digging."


def voice_id() -> str:
    return os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE


# --- pre-rendered fallback greeting ------------------------------------------

_fallback_audio: bytes | None = None


async def prerender_fallback_greeting() -> None:
    """Render the fallback line once at server start so it plays with no TTS delay."""
    global _fallback_audio
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id()}"
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                url, params={"output_format": f"pcm_{OUT_SAMPLE_RATE}"},
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                json={"text": FALLBACK_GREETING, "model_id": TTS_MODEL},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status != 200:
                    logger.warning(f"fallback greeting prerender failed: {r.status} {await r.text()}")
                    return
                _fallback_audio = await r.read()
        logger.info(f"pre-rendered fallback greeting ({len(_fallback_audio)} bytes)")
    except Exception as e:      # never block startup on this; we fall back to live TTS
        logger.warning(f"fallback greeting prerender failed: {e}")


# --- text cleaning before TTS (step 5a/5b) --------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?<![\w/])(?:[\w.-]+/)+([\w-]+)\.[A-Za-z]{1,5}\b|(?<![\w/])([\w-]+)\.(?:py|tsx?|go|rs|java|rb|json|toml|ya?ml)\b")


def _say_path(m: re.Match) -> str:
    return short(m.group(0))       # src/auth/token_service.py -> the token service


def _backticks(m: re.Match) -> str:
    inner = m.group(1)
    return inner if len(inner.split()) <= 4 else ""      # short identifiers stay, code goes


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", _backticks, text)
    text = _URL.sub("", text)
    text = _PATH.sub(_say_path, text)
    text = re.sub(r"^\s*(?:[-*•]|\d+\.)\s+", "", text, flags=re.M)     # list markers
    text = re.sub(r"\*{1,3}", "", text)                                # bold/italic
    text = re.sub(r"^\s*(?:#+|>)\s*", "", text, flags=re.M)            # headings/quotes
    text = re.sub(r"(?<=\w)_+(?=\w)", " ", text).replace("_", "")      # get_user -> get user
    text = text.replace("->", " to ").replace("=>", " to ").replace("()", "")
    return re.sub(r"\s{2,}", " ", text).strip()


# --- the agent processor ---------------------------------------------------------

@dataclasses.dataclass
class Speculation:
    text: str
    session: Session
    started: float
    task: asyncio.Task | None = None


def _same_words(a: str, b: str) -> bool:
    """Flux's eager transcript matches the final one unless the user resumed, but compare
    words only, so punctuation or casing differences don't throw the head start away."""
    norm = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    return norm(a) == norm(b)


HEARD_NOTE_CHARS = 300


class RepoAgent(FrameProcessor):
    """Runs answer() on each final user transcript and streams sentences to the TTS."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self.s = session
        self.scheduler = Scheduler(self._say_progress)
        self.heard: list[str] = []             # words played since the current answer began
        self._task: asyncio.Task | None = None
        self._side_tasks: set[asyncio.Task] = set()
        self._answer_open = False              # answer generating, queued, or still playing
        self._barge_note = ""                  # what the user heard before cutting in
        self._spec: Speculation | None = None  # step 5e: answer started on a draft transcript
        self._held: list[Frame] | None = None  # frames held until the draft is confirmed

    @property
    def user_talking(self) -> bool:
        return self.scheduler.user_talking

    @property
    def answering(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- frames --
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.spawn(self.scheduler.run())
        elif isinstance(frame, BotStartedSpeakingFrame):
            if direction == FrameDirection.UPSTREAM:
                self.scheduler.on_bot_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if direction == FrameDirection.UPSTREAM:
                self.scheduler.on_bot_stopped()
                if not self.answering:
                    self._answer_open = False  # the answer's audio has finished
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.scheduler.on_user_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.scheduler.on_user_stopped()
        elif isinstance(frame, InterruptionFrame):
            await self._cancel_speculation("user spoke")
            await self._barge_in()
        elif isinstance(frame, EagerTranscriptionFrame) and frame.text.strip():
            await self._speculate(frame.text.strip())
        elif isinstance(frame, UserTurnResponseCancelFrame):
            await self._cancel_speculation("turn resumed")
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self._on_final_transcript(frame.text.strip())
        await self.push_frame(frame, direction)

    def _question_for(self, text: str) -> str:
        return f"{self._barge_note}\n{text}" if self._barge_note else text

    async def _on_final_transcript(self, text: str):
        spec = self._spec
        if spec and _same_words(spec.text, text):
            await self._confirm_speculation(text)
            return
        await self._cancel_speculation("transcript changed" if spec else "")
        await self._cancel_answer()
        await emit(self.s, "user", text=text)
        question = self._question_for(text)
        self._barge_note = ""
        self.heard, self._answer_open = [], True
        logger.info(f"question: {question!r}")
        self._task = asyncio.create_task(self._answer(question, self.s), name="answer")

    # -- step 5e: eager end of turn --------------------------------------------------
    async def _speculate(self, text: str):
        """Flux thinks the turn has probably ended: start answering on a copy of the
        session, and hold everything it produces until the final transcript confirms it."""
        await self._cancel_speculation("new draft")
        sub = dataclasses.replace(self.s, history=list(self.s.history),
                                  focus=copy.deepcopy(self.s.focus), events=asyncio.Queue())
        self._held = []
        spec = Speculation(text=text, session=sub, started=time.monotonic())
        spec.task = asyncio.create_task(self._answer(self._question_for(text), sub), name="answer")
        self._spec = spec
        logger.info(f"eager: answering draft {text!r}")

    async def _confirm_speculation(self, text: str):
        spec, held = self._spec, self._held or []
        self._spec, self._held = None, None
        logger.info(f"eager: confirmed, {time.monotonic() - spec.started:.2f}s head start, "
                    f"{len(held)} frames released")
        await emit(self.s, "user", text=text)
        # Adopt the speculative turn: its history and focus become the session's. The task
        # keeps appending to the same list, and its events now go straight to the page.
        self.s.history, self.s.focus = spec.session.history, spec.session.focus
        while not spec.session.events.empty():
            await self.s.events.put(spec.session.events.get_nowait())
        spec.session.events = self.s.events
        self._barge_note = ""
        self.heard, self._answer_open = [], True
        self._task = spec.task
        for f in held:
            await self._release(f)

    async def _cancel_speculation(self, why: str):
        spec = self._spec
        if not spec:
            return
        self._spec, self._held = None, None
        spec.task.cancel()
        try:
            await spec.task
        except (asyncio.CancelledError, Exception):
            pass
        if why:
            logger.info(f"eager: discarded ({why})")

    # -- output gate -------------------------------------------------------------------
    def _is_live(self, session: Session) -> bool:
        """False only for a draft answer that hasn't been confirmed yet."""
        return not (self._spec is not None and session is self._spec.session)

    async def _out(self, frame: Frame, session: Session):
        """Answer output goes through here: held while its turn is only a draft."""
        if self._spec is not None and session is self._spec.session and self._held is not None:
            self._held.append(frame)
        elif self._spec is not None and session is not self.s and session is not self._spec.session:
            return      # a stale, cancelled speculation
        else:
            await self._release(frame)

    async def _release(self, frame: Frame):
        if isinstance(frame, LLMTextFrame):
            self.scheduler.speech_queued()
        await self.push_frame(frame)

    async def _barge_in(self):
        """Step 5d: the user started talking. Pipecat drops queued TTS audio; we stop the
        answer, drop any pending progress line, and remember what was actually heard."""
        self.scheduler.clear()
        if not self._answer_open:
            return
        heard = " ".join(w.strip() for w in self.heard if w.strip())
        if len(heard) > HEARD_NOTE_CHARS:
            heard = "…" + heard[-HEARD_NOTE_CHARS:]
        self._barge_note = (f'[The user interrupted your last answer. They heard only: "{heard}"]'
                            if heard else "[The user interrupted before you said anything.]")
        logger.info(f"barge-in after {len(heard)} chars heard")
        self._answer_open = False
        await self._cancel_answer()

    async def _cancel_answer(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    # -- answering --
    async def _answer(self, text: str, session: Session):
        buf = ""
        s = session

        async def speak(t: str):
            await self._out(LLMTextFrame(t), s)

        async def flush(final: bool = False):
            nonlocal buf
            parts = _SENTENCE_END.split(buf)
            done, buf = (parts, "") if final else (parts[:-1], parts[-1])
            for sentence in done:
                spoken = clean_for_speech(sentence)
                if spoken:
                    await speak(spoken + " ")

        async def on_text(t: str):
            nonlocal buf
            buf += t
            await flush()

        async def on_event(ev: dict):
            if ev["type"] != "tool_call":
                return
            # The round ended in tool calls: speak its preamble now ("Let me check X."),
            # don't hold it until the next round's text arrives.
            if buf.strip():
                await flush(final=True)
            pe = tool_event(ev["name"], ev["input"])
            await emit(s, "progress", text=pe["text"], tool=pe["tool"])   # screen: every event
            if self._is_live(s):
                self.scheduler.on_progress(pe)                            # voice: only in gaps

        await self._out(LLMFullResponseStartFrame(), s)
        try:
            if not s.cloned.is_set():
                await emit(s, "progress", text="Waiting for the clone to finish…")
                await asyncio.wait_for(s.cloned.wait(), CLONE_WAIT_S)
            stats = await answer(s, text, on_text, on_event)
            await flush(final=True)
            logger.info(f"answer: highlight={stats.highlight} rounds={stats.rounds} tools={len(stats.tool_calls)} "
                        f"first_text={stats.first_text_s} total={stats.total_s:.2f}s")
            await emit(s, "answer_done", rounds=stats.rounds, tools=len(stats.tool_calls),
                       highlight=stats.highlight, first_text_s=stats.first_text_s, total_s=stats.total_s,
                       input_tokens=stats.input_tokens, cache_read=stats.cache_read,
                       output_tokens=stats.output_tokens)
        except asyncio.TimeoutError:
            await speak("Sorry, I couldn't get the repo cloned, so I can't look that up.")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("answer failed")
            await emit(s, "error", text=str(e))
            await speak("Sorry, something went wrong while I was looking that up.")
        finally:
            if self._is_live(s):
                self.scheduler.pending = None     # the answer is in; progress lines are moot
        await self._out(LLMFullResponseEndFrame(), s)

    # -- speaking outside an answer (greeting, progress, setup updates) --
    async def say(self, text: str):
        self.scheduler.speech_queued()
        await self.push_frame(TTSSpeakFrame(text))

    async def _say_progress(self, text: str):
        await self.say(text)

    async def play_pcm(self, pcm: bytes, chunk_ms: int = 20):
        self.scheduler.playing_for(len(pcm) / (OUT_SAMPLE_RATE * 2))
        step = OUT_SAMPLE_RATE * 2 * chunk_ms // 1000          # 16-bit mono
        for i in range(0, len(pcm), step):
            await self.push_frame(OutputAudioRawFrame(pcm[i:i + step], OUT_SAMPLE_RATE, 1))

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self._side_tasks.add(t)
        t.add_done_callback(self._side_tasks.discard)

    async def cleanup(self):
        await super().cleanup()
        await self._cancel_speculation("")
        await self._cancel_answer()
        for t in list(self._side_tasks):
            t.cancel()


class HeardTracker(FrameProcessor):
    """Sits after the output transport, which releases word frames as their audio plays,
    so what reaches here is what the user has actually heard."""

    def __init__(self, agent: RepoAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and direction == FrameDirection.DOWNSTREAM:
            self.agent.heard.append(frame.text)
        await self.push_frame(frame, direction)


# --- step 2: informed greeting ------------------------------------------------

async def greet(agent: RepoAgent):
    s = agent.s
    try:
        await asyncio.wait_for(s.ready.wait(), GREETING_WAIT_S)
        if s.error:
            await agent.say("Sorry, I couldn't open that repo. Check the link and try again.")
        elif s.overview:
            await agent.say(f"Hi! {s.overview} What do you want to know?")
        else:
            await agent.say("Hi! I've got the repo open. What do you want to know?")
        return
    except asyncio.TimeoutError:
        pass
    if _fallback_audio:
        await agent.play_pcm(_fallback_audio)
    else:
        await agent.say(FALLBACK_GREETING)
    agent.spawn(speak_overview_when_ready(agent))


OVERVIEW_GAP_S = Scheduler.GAP_S     # same silence rule as other progress lines
OVERVIEW_GIVE_UP_S = 30


async def speak_overview_when_ready(agent: RepoAgent):
    """Setup finished after the fallback greeting: the overview is a progress update, so it
    goes through the same gap rules, and is skipped once the user has started asking."""
    s = agent.s
    await s.ready.wait()
    if s.error or not s.overview:
        return
    deadline = asyncio.get_running_loop().time() + OVERVIEW_GIVE_UP_S
    while asyncio.get_running_loop().time() < deadline:
        if s.history or agent.answering:
            return
        if agent.scheduler.can_speak() and agent.scheduler.quiet_for() > OVERVIEW_GAP_S:
            await agent.say(s.overview)
            return
        await asyncio.sleep(Scheduler.TICK_S)


# --- pipeline -----------------------------------------------------------------------

class SentenceFlushingTTS(ElevenLabsTTSService):
    """ElevenLabs in TOKEN mode, flushed after every complete sentence.

    RepoAgent sends whole sentences. In TOKEN mode they reach ElevenLabs at once, but its
    WebSocket buffers short text before generating (~1 s TTFB for a one-line preamble). A
    flush after each sentence starts audio right away; the context stays open, so the rest
    of the answer keeps its prosody.
    """

    async def run_tts(self, text: str, context_id: str):
        async for frame in super().run_tts(text, context_id):
            yield frame
        if text.rstrip().endswith((".", "!", "?", "…")):
            await self.flush_audio(context_id=context_id)



async def run_bot(connection: SmallWebRTCConnection, s: Session):
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = DeepgramFluxSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        enable_eager_end_of_turn=EAGER_END_OF_TURN,
        settings=DeepgramFluxSTTService.Settings(eager_eot_threshold=EAGER_EOT_THRESHOLD)
        if EAGER_END_OF_TURN else None,
    )
    turns = UserTurnProcessor(user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=True))
    agent = RepoAgent(s)
    tts = SentenceFlushingTTS(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice=voice_id(), model=TTS_MODEL),
        # RepoAgent already sends whole, cleaned sentences. SENTENCE mode would hold each one
        # until the next sentence starts, so the "Let me check X." preamble would wait for
        # the tool calls to finish.
        text_aggregation_mode=TextAggregationMode.TOKEN,
    )
    pipeline = Pipeline([transport.input(), stt, turns, agent, tts, transport.output(),
                         HeardTracker(agent)])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=OUT_SAMPLE_RATE,
                              enable_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info(f"session {s.id}: client connected")
        agent.spawn(greet(agent))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"session {s.id}: client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
