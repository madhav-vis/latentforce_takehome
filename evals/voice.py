"""Voice latency harness: speak the eval questions into the real pipeline over WebRTC.

    python -m evals.voice                       # starts a server, 6 questions per repo
    python -m evals.voice --n 3 --repo https://github.com/pallets/flask
    python -m evals.voice --url http://127.0.0.1:8000   # use a server you started

Each question is synthesized once with ElevenLabs (a different voice from the bot) and
cached in evals/audio/. A small aiortc client plays it as the "mic", the same way the page
does, and timestamps on the client's clock:

  speech_end    last voiced sample of the question leaves the client
  transcript    the final transcript arrives over SSE (end of turn decided)
  first_audio   first audible bot frame after speech_end
  audio_end     bot audio has been quiet for QUIET_S

Reported: end_of_turn = transcript - speech_end; to_first_audio = first_audio - speech_end
(the number that matters, target p50 ~1 s / p95 < 2 s); after_transcript = first_audio -
transcript; answer_audio = audio_end - first_audio. Run it from the region you deploy to.
"""

import argparse
import asyncio
import fractions
import hashlib
import json
import os
import socket
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
from dotenv import load_dotenv

HERE = Path(__file__).parent
ROOT = HERE.parent
AUDIO_DIR = HERE / "audio"
RATE = 16000
FRAME = 320                      # 20 ms
QUESTION_VOICE = "pNInz6obpgDQGcFmaJgB"     # ElevenLabs premade "Adam" (the bot is "Rachel")
LOUD_RMS = 300                   # bot frame counts as audible above this
QUIET_S = 1.5                    # bot audio "ended" after this much quiet
VOICED = 500                     # question sample counts as speech above this


# --- question audio -------------------------------------------------------------

async def synthesize(text: str) -> bytes:
    """16 kHz mono PCM of the question, cached on disk."""
    key = hashlib.sha1(f"{QUESTION_VOICE}|{text}".encode()).hexdigest()[:16]
    path = AUDIO_DIR / f"{key}.pcm"
    if path.exists():
        return path.read_bytes()
    AUDIO_DIR.mkdir(exist_ok=True)
    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{QUESTION_VOICE}",
            params={"output_format": f"pcm_{RATE}"},
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            json={"text": text, "model_id": "eleven_flash_v2_5"},
        ) as r:
            if r.status != 200:
                raise RuntimeError(f"ElevenLabs {r.status}: {await r.text()}")
            pcm = await r.read()
    path.write_bytes(pcm)
    return pcm


def voiced_span(pcm: bytes) -> tuple[int, int]:
    """(first, last) voiced sample index."""
    a = np.abs(np.frombuffer(pcm, np.int16))
    idx = np.nonzero(a > VOICED)[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else (0, len(a))


# --- client media ---------------------------------------------------------------

class Mic(MediaStreamTrack):
    """Plays queued PCM in real time; silence otherwise. Records when speech ends."""
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.buf = bytearray()
        self.pts = 0
        self.t0: float | None = None
        self.speech_end_at: float | None = None   # client time the last voiced sample is sent
        self._end_sample: int | None = None        # absolute sample index of that sample

    def say(self, pcm: bytes):
        _, last = voiced_span(pcm)
        self._end_sample = self.pts + len(self.buf) // 2 + last
        self.speech_end_at = None
        self.buf += pcm

    async def recv(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        await asyncio.sleep(max(0, self.t0 + self.pts / RATE - time.monotonic()))
        chunk = bytes(self.buf[:2 * FRAME]).ljust(2 * FRAME, b"\0")
        del self.buf[:2 * FRAME]
        if self._end_sample is not None and self.pts <= self._end_sample < self.pts + FRAME:
            self.speech_end_at = time.monotonic()
        f = AudioFrame.from_ndarray(np.frombuffer(chunk, np.int16).reshape(1, -1), format="s16", layout="mono")
        f.sample_rate, f.pts, f.time_base = RATE, self.pts, fractions.Fraction(1, RATE)
        self.pts += FRAME
        return f


class Ear:
    """Watches bot audio: when it becomes audible and when it goes quiet."""

    def __init__(self):
        self.loud_at: list[float] = []      # start of each audible stretch
        self.last_loud = 0.0
        self.talking = False

    async def listen(self, track):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            rms = float(np.sqrt(np.mean(frame.to_ndarray().astype(np.float32) ** 2)))
            now = time.monotonic()
            if rms > LOUD_RMS:
                if not self.talking:
                    self.loud_at.append(now)
                self.talking, self.last_loud = True, now
            elif self.talking and now - self.last_loud > 0.3:
                self.talking = False

    def first_after(self, t: float) -> float | None:
        return next((x for x in self.loud_at if x > t), None)

    async def wait_quiet(self, since: float, timeout: float) -> float | None:
        """Wait until bot audio has been quiet QUIET_S after `since`; return last loud time."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self.talking and time.monotonic() - max(self.last_loud, since) > QUIET_S:
                return self.last_loud if self.last_loud > since else None
            await asyncio.sleep(0.05)
        return None


# --- one session ---------------------------------------------------------------

async def run_repo(base: str, repo: str, questions: list[dict], log) -> list[dict]:
    mic, ear = Mic(), Ear()
    pc = RTCPeerConnection()
    pc.addTrack(mic)
    pc.createDataChannel("chat")
    pc.on("track", lambda tr: asyncio.ensure_future(ear.listen(tr)) if tr.kind == "audio" else None)
    events: list[tuple[float, dict]] = []
    rows = []
    async with aiohttp.ClientSession() as http:
        r = await http.post(f"{base}/session", json={"repo_url": repo})
        r.raise_for_status()
        sid = (await r.json())["session_id"]

        async def sse():
            async with http.get(f"{base}/events/{sid}", timeout=aiohttp.ClientTimeout(total=None)) as resp:
                async for line in resp.content:
                    if line.startswith(b"data: "):
                        events.append((time.monotonic(), json.loads(line[6:])))
        sse_task = asyncio.create_task(sse())

        await pc.setLocalDescription(await pc.createOffer())
        r = await http.post(f"{base}/offer?session_id={sid}",
                            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        ans = await r.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        connected = time.monotonic()
        while not ear.loud_at and time.monotonic() - connected < 15:     # greeting starts
            await asyncio.sleep(0.05)
        greet_end = await ear.wait_quiet(connected, timeout=60)
        log(f"  greeting: {'%.1fs of audio' % (greet_end - ear.loud_at[0]) if greet_end and ear.loud_at else 'none heard'}")

        for q in questions:
            pcm = await synthesize(q["question"])
            asked = time.monotonic()
            mic.say(pcm)
            while mic.speech_end_at is None and time.monotonic() - asked < 30:
                await asyncio.sleep(0.01)
            speech_end = mic.speech_end_at or asked
            # Tool calls leave silent gaps mid-answer, so wait for the server's answer_done
            # before deciding the bot has finished talking.
            done = None
            while done is None and time.monotonic() - speech_end < 90:
                done = next(((t, e) for t, e in events if t > asked and e["type"] == "answer_done"), None)
                await asyncio.sleep(0.05)
            audio_end = await ear.wait_quiet(done[0] if done else speech_end, timeout=120)
            first_audio = ear.first_after(speech_end)
            transcript = next(((t, e["text"]) for t, e in events if t > asked and e["type"] == "user"), None)
            shown = [e for t, e in events if t > asked and e["type"] == "show_code"]
            row = {
                "id": q["id"], "repo": repo, "question": q["question"],
                "heard_as": transcript[1] if transcript else None,
                "end_of_turn": (transcript[0] - speech_end) if transcript else None,
                "to_first_audio": (first_audio - speech_end) if first_audio else None,
                "after_transcript": (first_audio - transcript[0]) if first_audio and transcript else None,
                "answer_audio": (audio_end - first_audio) if audio_end and first_audio else None,
                "server": {k: v for k, v in done[1].items() if k != "type"} if done else None,
                "first_highlight": ((next(t for t, e in events if t > asked and e["type"] == "show_code") - speech_end)
                                    if shown else None),
            }
            rows.append(row)
            fmt = lambda x: f"{x:5.2f}s" if x is not None else "   -  "
            log(f"  {q['id']:<9} eot {fmt(row['end_of_turn'])}  first audio {fmt(row['to_first_audio'])}"
                f"  (after transcript {fmt(row['after_transcript'])})  answer {fmt(row['answer_audio'])}"
                f"  {q['question'][:40]}")
            await asyncio.sleep(1.0)
        sse_task.cancel()
    await pc.close()
    return rows


# --- server ---------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def start_server(log_path: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "server.main"], cwd=ROOT,
                            env={**os.environ, "PORT": str(port)},
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as http:
        for _ in range(120):
            try:
                async with http.get(base + "/") as r:
                    if r.status == 200:
                        return proc, base
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"server didn't start; see {log_path}")


def pctl(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows), "no_audio": sum(r["to_first_audio"] is None for r in rows)}
    for k in ("end_of_turn", "to_first_audio", "after_transcript", "answer_audio", "first_highlight"):
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = {"p50": pctl(vals, .5), "p95": pctl(vals, .95), "mean": st.mean(vals) if vals else None}
    return out


async def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="use a running server instead of starting one")
    ap.add_argument("--repo")
    ap.add_argument("--ids")
    ap.add_argument("--n", type=int, default=6, help="questions per repo (non-follow-up)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    qs = [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines() if l.strip()]
    qs = [q for q in qs if q["bucket"] != "follow-up"]
    if args.repo:
        qs = [q for q in qs if q["repo"].rstrip("/") == args.repo.rstrip("/")]
    if args.ids:
        want = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    by_repo: dict[str, list[dict]] = {}
    for q in qs:
        by_repo.setdefault(q["repo"], [])
        if args.ids or len(by_repo[q["repo"]]) < args.n:
            by_repo[q["repo"]].append(q)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{args.label}" if args.label else "")
    proc = None
    if args.url:
        base = args.url.rstrip("/")
    else:
        proc, base = await start_server(out_dir / f"voice-{stamp}.server.log")
    t0 = time.monotonic()
    log = lambda m: print(f"{time.monotonic() - t0:7.1f}s {m}", flush=True)
    rows = []
    try:
        for repo, rq in by_repo.items():
            log(repo)
            rows += await run_repo(base, repo, rq, log)
    finally:
        if proc:
            proc.terminate()
            proc.wait(10)
    (out_dir / f"voice-{stamp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = summarize(rows)
    (out_dir / f"voice-{stamp}.summary.json").write_text(json.dumps(s, indent=2))
    print(f"\n{'':18}{'p50':>8}{'p95':>8}{'mean':>8}")
    for k in ("end_of_turn", "to_first_audio", "after_transcript", "answer_audio", "first_highlight"):
        v = s[k]
        f = lambda x: f"{x:7.2f}s" if x is not None else "      - "
        print(f"{k:<18}{f(v['p50'])}{f(v['p95'])}{f(v['mean'])}")
    print(f"n={s['n']}  no audio={s['no_audio']}   (target: to_first_audio p50 ~1 s, p95 < 2 s)")
    print(f"rows: {out_dir / f'voice-{stamp}.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
