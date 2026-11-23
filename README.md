# Voice codebase explainer

Paste a public GitHub link and talk to the repo. The agent answers out loud in a few sentences while the page highlights the code it's talking about.

- **Agent:** Claude Haiku 5.5, with tools to grep, read files and show code. Long multi-file traces can be handed to Sonnet 5.5.
- **Speech:** Deepgram Flux for speech-to-text and ElevenLabs Flash v2.5 for text-to-speech.
- **Plumbing:** Pipecat over SmallWebRTC, FastAPI with server-sent events (SSE), and a single HTML page.

There's no database and no vector index. Each session clones the repo, builds an Aider-style repo map in memory, and the agent searches the real files with ripgrep.

**Live demo: https://78-47-65-212.sslip.io**. Ask me for the passcode.

## What it adds over a basic voice agent

A basic voice agent transcribes your question, sends it to a model with a few file tools, waits for the full reply, then reads it aloud. Here is what this one does differently. The numbers come from the evals described under [Results](#results), comparing against the first working version of this app.

**Faster to start talking**
- **Short spoken preamble before the first lookup** ("Let me check the router"), so you hear something while the tools run. A typical answer needs 2–3 s of lookups.
- **Sentence-by-sentence text-to-speech.** Each sentence is flushed to ElevenLabs as soon as it's complete instead of waiting for the whole answer. Time to the first synthesized audio dropped from 1.0–1.4 s to about 0.2 s.
- **Eager end-of-turn.** It starts answering on a draft transcript before Flux confirms you've stopped, and holds the audio until the transcript is confirmed. If you were still talking, the draft is thrown away.
- **Thinking off and prompt caching** on the voice turns. Turning thinking off cut the model's first-text time from 0.97 s to 0.40 s (p50), and caching the conversation history cut input cost by 57%.
- **Overall:** from the end of your speech to the first sound, 1.44 s p50 and 1.64 s p95, down from 2.51 s and 2.90 s.

**Knows where to look**
- **Repo map in the prompt.** A tree-sitter symbol map, ranked with PageRank and shown with line numbers, lets the agent go straight to the right file. Mean tool rounds dropped from 3.28 to 2.69 with the map, compared with no map.
- **Parallel tool calls.** It batches greps and file reads into one round.
- **Optional Sonnet 5.5 deep dive** for "trace the whole flow" questions. It was used on 2–3 of 40 eval questions, with no measurable change in correctness.

**Shows what it's talking about**
- **Synced code pane.** Every answer highlights the file and lines being discussed. If the model forgets to, the app falls back to what the model read or named. The right file was on screen for 97–100% of answers, up from 83%.
- **Spoken progress in long silences.** Tool calls show on screen as they happen, and when a lookup runs long, a short spoken update ("Reading the session module…") fills the silence.

**Behaves like a conversation**
- **Follow-ups resolve.** What's on screen (the file, the symbol, and its callers and callees) is passed along with the next question, so "what calls that?" works. The judge rated 100% of follow-ups as answered without a clarifying question.
- **Barge-in.** You can interrupt. Speech stops immediately, and the agent is told exactly which words you heard before you cut in.
- **Speakable answers.** File paths and identifiers are rewritten into words before speech. Paths read aloud dropped from 18% of answers to 8%.
- **No wait on the first visit.** A greeting plays as soon as you connect, while the repo clones and the map builds. A short spoken overview of the repo follows when it's ready. Maps and overviews are cached per commit.

**Accuracy:** correctness rose from 3.88 to 4.03 and 4.10 across two runs, scored 1–5 by a Sonnet 5.5 judge on 40 questions over two repos.

Not measured by the evals: barge-in, the spoken progress updates and the greeting were tested by hand only.

## Run it

You need three API keys:

| Key | Notes |
|---|---|
| `ANTHROPIC_API_KEY` | |
| `DEEPGRAM_API_KEY` | |
| `ELEVENLABS_API_KEY` | Must have the **Text to Speech** permission |

```sh
cp .env.example .env        # then fill in the three keys
```

**Docker**

```sh
docker compose up --build
```

Open http://localhost:8000, paste a repo link and allow the microphone.

WebRTC audio travels over UDP directly between the browser and the container. If the page loads but the voice never connects, switch the service to host networking (the comment in `docker-compose.yml` shows how).

**Without Docker** (Python 3.12, `git`, and `ripgrep` on your PATH)

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m server.main     # http://localhost:8000
```

**Text-only CLI** (no audio keys needed)

```sh
.venv/bin/python -m server.cli https://github.com/pallets/flask
```

Ctrl-C during an answer cancels it.

**Things to try**

- "What does this project do?"
- "Where are routes registered?" then "and who calls that?" (follow-ups use what's on screen).
- "Walk me through everything that happens when a request comes in." This triggers the Sonnet deep dive and takes about 15 seconds.
- Start talking while it's answering. It stops, and it knows how much of the answer you heard.

## Deploy

To put it online for a few people on your own keys, use a small VPS with Docker and Caddy.

**Pick a host with UDP.** WebRTC audio needs a long-running server that accepts UDP, so serverless hosts won't work. A Hetzner CX22 or a DigitalOcean 2 GB droplet running Ubuntu 24.04 is plenty. Avoid AWS Lightsail and EC2, which put the public IP behind NAT.

1. Install Docker on the server:

   ```sh
   ssh root@<server-ip>
   curl -fsSL https://get.docker.com | sh
   ```

2. Get the code and create `.env`:

   ```sh
   git clone https://github.com/madhav-vis/latentforce_takehome && cd latentforce_takehome
   cp .env.example .env
   ```

   Fill in `.env`:
   - the three API keys;
   - `ACCESS_CODE`, a passcode the page asks for once per browser;
   - `SITE_HOST`, the server IP with dashes plus `.sslip.io`, for example `203-0-113-5.sslip.io`. sslip.io resolves that name to the IP, so you don't need a domain.

3. Start it:

   ```sh
   docker compose -f docker-compose.prod.yml up -d --build
   ```

4. Open `https://<SITE_HOST>`. Caddy gets a Let's Encrypt certificate on the first request. Browsers only allow the microphone over HTTPS.

**Firewall.** If you add one, allow TCP 22, 80 and 443, plus UDP 1024–65535 for WebRTC audio.

**Spending limits.** Set them in the Anthropic, Deepgram and ElevenLabs consoles as a backstop.

**Updates and logs:**

```sh
git pull && docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml logs -f
```

## How a session works

1. **Link pasted** (`POST /session`). The server validates the URL, then starts a shallow clone and setup while the browser asks for the microphone. Setup:
   - lists the source files, skipping vendored code, build output, lockfiles and binaries;
   - builds a tree-sitter tag map ranked by PageRank;
   - writes a three-sentence overview with one Haiku call.

   Results are cached per commit in `cache/`.
2. **Greeting.** A pre-rendered generic line plays the moment WebRTC connects. The repo overview follows as soon as it's ready.
3. **Question.** Flux decides when you've finished speaking. The agent has the repo map in its system prompt and calls `grep`, `read_file`, `list_dir` and `show_code`, in parallel when it can. It speaks a short preamble ("Let me check the router") before its first lookup.
4. **Progress.** Every tool call appears on screen. When a lookup runs long and nothing else is being said, a short spoken update ("Reading the session module…") fills the gap.
5. **Answer.** Text streams into ElevenLabs one sentence at a time. A cleaner rewrites file paths and identifiers into speakable words. `show_code` sends the file and line range to the code pane over SSE.
6. **Follow-ups and barge-in.** The last thing shown (file, symbol, and its callers and callees from the map) is passed along with the next question, so "what calls that?" resolves. If you interrupt, the agent is told exactly which words you heard.

Two latency features sit on top:

- **Eager end-of-turn.** When Flux predicts you've finished speaking, the agent starts answering on a copy of the session with its audio held back. If Flux confirms the same words, the held answer is released. If you keep talking, the draft is thrown away.
- **Deep dive.** For "trace the whole flow" questions, Haiku says it needs a few seconds and calls Sonnet 5.5 with the same tools. Sonnet's lookups and highlights stream to the screen while it works.

### Code map

| Path | What it does |
|---|---|
| [server/main.py](server/main.py) | FastAPI routes: `/session`, `/events/{id}` (SSE), `/offer` (WebRTC), `/` |
| [server/setup.py](server/setup.py) | Clone, file listing, overview, per-commit cache |
| [server/repomap.py](server/repomap.py) | tree-sitter tags, reference graph, PageRank, map rendering |
| [server/agent.py](server/agent.py) | Streaming Claude tool loop, prompts, Sonnet deep dive |
| [server/tools.py](server/tools.py) | Sandboxed tools, focus tracking, highlight fallbacks |
| [server/narrator.py](server/narrator.py) | Turns tool calls into screen text and spoken updates; decides when it may speak |
| [server/pipeline.py](server/pipeline.py) | Pipecat pipeline, eager end-of-turn, barge-in, speech cleanup |
| [web/index.html](web/index.html) | The page: link box, voice bubble, status line, code pane |
| [evals/](evals/) | Question set, text harness, map ablation, voice latency harness |

## Results

All measured on two pinned repos (Flask, and FastAPI's full-stack template), 20 questions each, from a laptop on home internet.

**Voice latency.** "First audio" is the time from the end of your speech to the first sound of the answer. Measured with `evals/voice.py` (12 spoken questions per run).

| Configuration | First audio p50 | First audio p95 |
|---|---|---|
| First working pipeline | 2.51 s | 2.90 s |
| Thinking off, prompt caching, sentence-level text-to-speech flushing | 1.82 s | 2.06 s |
| Plus eager end-of-turn at threshold 0.3 (current) | **1.44 s** | **1.64 s** |

The first highlight appears at about 1.9 s (p50). The target was about 1 s p50 and under 2 s p95, so p95 is met and p50 is not. Most of the remaining time is Flux's end-of-turn decision (about 0.6 s) plus Haiku's first tokens and ElevenLabs' first audio.

**Answer quality.** Measured with `evals/run.py` on 40 questions. A Sonnet 5.5 judge scores each answer 1–5 against a written reference.

| Metric | First baseline | Current (two runs) |
|---|---|---|
| Correctness (1–5) | 3.88 | 4.03 / 4.10 |
| Answers scored 4 or 5 | 65% | 82% / 82% |
| Highlighted the right file | 83% | 97% / 100% |
| Model rounds (median) | 3 | 2 |
| Time to first text (model, p50) | 1.01 s | 0.40 s |
| Agent cost per question | $0.00075 | $0.00048 |

The cost row counts Haiku tokens only. It excludes speech-to-text, text-to-speech and Sonnet deep dives.

**Repo map ablation** (`evals/ablate.py`). Putting the map in the prompt cut the mean number of model rounds from 3.28 to 2.69, because the agent can jump straight to the right file and line instead of searching first.

**Deep dive.** Haiku used it on 2–3 of the 40 questions. Correctness with it on (4.17 and 3.83 across two runs) was within run-to-run noise of having it off. Those answers take 13–19 s instead of 2–3 s. It's on by default because it only triggers on explicit whole-flow questions; set `DEEP_DIVE = False` in [server/agent.py](server/agent.py) to disable it.

### Running the evals

```sh
.venv/bin/python -m evals.run                    # text eval, all 40 questions
.venv/bin/python -m evals.run --repeats 2 --label mytest
.venv/bin/python -m evals.ablate                 # map vs no map
.venv/bin/python -m evals.voice                  # spoken latency; starts its own server
```

Results are written to `evals/results/`. The voice harness synthesizes the questions with a different ElevenLabs voice and caches the audio in `evals/audio/`.

## Configuration

Environment variables, all optional except the three keys:

| Variable | Default | Effect |
|---|---|---|
| `ELEVENLABS_VOICE_ID` | Rachel (`21m00Tcm4TlvDq8ikWAM`) | Bot voice |
| `EAGER_EOT` | `1` | `0` turns eager end-of-turn off |
| `EAGER_EOT_THRESHOLD` | `0.3` | Flux eager threshold. Higher means fewer wasted drafts but less head start. |
| `ACCESS_CODE` | unset | Passcode required to start a session; unset means no gate |
| `SITE_HOST` | unset | Hostname Caddy serves in `docker-compose.prod.yml` |
| `MAX_SESSIONS` | `3` | Concurrent sessions before `/session` returns 429 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Server bind address (the Docker image uses `0.0.0.0`) |
| `REPOS_DIR` | `/tmp/repos` | Where clones go; each is deleted when its session ends |
| `CACHE_DIR` | `./cache` | Per-commit repo maps and overviews |

Model, effort, token limits and the deep-dive settings are constants at the top of [server/agent.py](server/agent.py).

## Limits

- **Public GitHub repos only.** Private repos would need a GitHub token, which v1 doesn't support. The clone runs with prompts disabled, so a private link fails fast with an error instead of hanging.
- **Repo size.** Files over about 1 MB are skipped, and indexing stops after about 5,000 source files. The clone times out after 30 seconds.
- **Language coverage.** The repo map works for languages whose tree-sitter grammar ships a tags query, which covers Python, JS/TS, Go, Rust, Java, C/C++, C#, Ruby, PHP, Swift, Kotlin and others. Other files can still be grepped and read; they just don't appear in the map.
- **Browser.** Developed against Chrome. Echo cancellation is whatever the browser's microphone provides, so headphones help on laptop speakers.
- **Light access control.** A deployed instance is protected only by the shared `ACCESS_CODE`, so keep spending limits set.
