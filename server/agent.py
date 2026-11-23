"""Step 3: a plain Claude tool-use loop.

Stream the response, run any tool calls in parallel, send all results back in one
message, repeat until the model answers without tools.
"""

import asyncio
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import anthropic

from .session import Session
from .tools import TOOLS, locate_mentioned_code, render_focus, run_tool

MODEL = "claude-haiku-5-5"
EFFORT = "low"          # Haiku 5.5 defaults to medium; low for voice turns
# Stage 8 A/B (2 runs each, 40 questions): thinking off cut time to first text from ~1.0 s to
# ~0.4 s (p95 1.7 s -> 0.55 s) for a correctness dip of ~0.15 (4.04 -> ~3.88). Allowed at low
# effort on Haiku 5.5. Pass thinking="adaptive" where depth matters more than speed.
THINKING = "off"
# Cache the conversation, not just the system prompt: uncached input grew to 10-15k tokens
# by the sixth voice turn, re-read on every tool round.
CACHE_HISTORY = True

# --- step 3d: the deep-dive helper --------------------------------------------------
DEEP_DIVE = True
DEEP_MODEL = "claude-sonnet-5-5"
DEEP_EFFORT = "medium"          # Sonnet 5.5 guidance: medium for multi-step tool use
DEEP_MAX_ROUNDS = 12
DEEP_MAX_TOKENS = 4000
DEEP_TIMEOUT_S = 75
DEEP_RESULT_CHARS = 4000
# Sonnet 5.5 can decline a request; the server-side fallback re-runs it on another model.
DEEP_BETA = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

DEEP_DIVE_RULE = """\
- If a question needs tracing a flow across many files ("walk me through what happens when
  a user signs up", "trace the whole request path"), say "Let me trace that, give me a few
  seconds." and call deep_dive instead of doing many lookups yourself. When it returns,
  answer from its notes as usual (calling show_code for the key lines).
"""

DEEP_DIVE_TOOL = {
    "name": "deep_dive",
    "description": ("Hand a question that needs tracing across many files to a slower, more "
                    "thorough helper with the same tools. Takes 10-40 seconds. Returns written "
                    "notes naming the files and functions involved, in order."),
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The full question, with any context from the conversation"}},
        "required": ["question"],
    },
}

DEEP_PROMPT = """\
You are a code-tracing helper for a voice assistant that explains the {repo} codebase. The
assistant handed you a question that needs following a flow across several files.

- Use the tools to read the real code. Batch independent lookups in one response.
- Use the repo map below to go straight to files; grep for names it doesn't list.
- Call show_code once or twice for the most important lines; the user's screen shows them.
- Then write notes for the assistant (not for the user): the steps in order, each naming the
  file path and function, in under 200 words. Plain text, no markdown headings.
- Only state what you read. Say what you couldn't confirm.

REPO MAP (most important files first)
{repo_map}
"""


async def run_deep_dive(s: Session, question: str, on_event: "EventCB") -> str:
    """Run the question on Sonnet with the same tools in a separate conversation. It shares
    the session's focus and events, so its highlights and progress reach the screen."""
    sub = dataclasses.replace(s, history=[])
    notes: list[str] = []

    async def collect(t: str):
        notes.append(t)

    await on_event({"type": "deep_dive_start", "question": question, "ts": time.monotonic()})
    try:
        st = await asyncio.wait_for(answer(
            sub, question, collect, on_event, model=DEEP_MODEL, effort=DEEP_EFFORT,
            thinking="adaptive", max_rounds=DEEP_MAX_ROUNDS, max_tokens=DEEP_MAX_TOKENS,
            tools=TOOLS, deep_dive=False, beta=DEEP_BETA,
            system_prompt=DEEP_PROMPT.format(repo=s.repo_name or "this", repo_map=s.repo_map or "(none yet)"),
        ), DEEP_TIMEOUT_S)
        text = "".join(notes).strip()
        log_note = f"[deep dive: {st.rounds} rounds, {len(st.tool_calls)} tool calls, {st.total_s:.1f}s]"
    except asyncio.TimeoutError:
        text = "".join(notes).strip()
        log_note = f"[deep dive timed out after {DEEP_TIMEOUT_S}s; partial notes]"
    await on_event({"type": "deep_dive_done", "ts": time.monotonic()})
    return f"{log_note}\n{text[:DEEP_RESULT_CHARS] or '(no notes)'}"
MAX_TOKENS = 600        # spoken answers are short; tune in the latency stage
MAX_ROUNDS = 6
MAX_JSON_RETRIES = 2
AUTO_SHOW_LINES = 60    # fallback highlight window when a turn read code but showed none

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


LENGTH_RULES = {
    # stage 1-7 wording
    "default": "- Answer in 1-3 short sentences, then offer to go deeper.",
    # stage 8 candidate: answers were running 70-80 words (~30 s of speech)
    "short": (
        "- Keep each spoken answer to about 40 words: lead with the direct answer in one\n"
        "  sentence, add at most two short sentences of detail, then stop. The screen shows the\n"
        "  code, so don't narrate it line by line. A preamble like \"Let me check X\" counts\n"
        "  toward the 40 words. End with at most a five-word offer, such as \"Want the details?\""
    ),
}
LENGTH_VARIANT = "default"

STABLE_PROMPT = """\
You explain the {repo} codebase out loud to a developer.

VOICE RULES
- Your words are spoken. No markdown, lists, code, or file paths read in full.
{length_rule}
- Say "the login handler in the auth service", not "src/auth/service.py line 42".
- Every answer about the code calls show_code for the lines you're talking about, even if
  they were on screen before: the screen is how the user follows along. Call it in the same
  response as your answer, after the text, never in a response of its own. Skip it only when
  no code is relevant.

SPEED RULES
- If you need tools, FIRST say one short sentence of what you'll check
  ("Let me look at the auth service."), THEN call tools.
- Batch lookups: call several tools at once when they don't depend on each other.
- The repo map lists key definitions with their line numbers. When it shows what you need,
  read_file a window around that line straight away. Grep only when the map doesn't say.
- Never read more than you need. Prefer 60-line windows around a definition.
{deep_dive_rule}
FOLLOW-UPS
- Resolve "that", "it", "there" and "this" from CURRENT FOCUS instead of asking the user.
  "Where's that called?" means: grep for the on-screen symbol.
- The files that use the focused file are the likely next reads.
- Follow-ups need show_code too: re-show the lines you're now talking about.

GROUNDING
- Only state what you read in this repo. If unsure, say so and offer to check.
"""


def build_stable_prompt(s: Session, variant: str | None = None, deep_dive: bool = True) -> str:
    """Identical every turn, so it sits before the cache breakpoint."""
    variant = variant or LENGTH_VARIANT
    parts = [STABLE_PROMPT.format(repo=s.repo_name or "this", length_rule=LENGTH_RULES[variant],
                                  deep_dive_rule=DEEP_DIVE_RULE if deep_dive else "")]
    if s.overview:
        parts.append(f"REPO OVERVIEW\n{s.overview}\n")
    if s.repo_map:
        parts.append(f"REPO MAP (most important files first)\n{s.repo_map}\n")
    else:
        parts.append("No repo map yet: start with list_dir('.') or grep to orient yourself.\n")
    return "\n".join(parts)


@dataclass
class TurnStats:
    rounds: int = 0
    tool_calls: list = field(default_factory=list)      # [(name, input)]
    shown: list = field(default_factory=list)           # show_code paths (model's own calls)
    auto_shown: str | None = None                       # fallback highlight, see AUTO_SHOW_LINES
    highlight: str = "none"     # where the screen update came from: model | read-fallback | text-fallback | none
    input_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int = 0
    stop_reasons: list = field(default_factory=list)
    first_text_s: float | None = None
    total_s: float = 0.0
    text: str = ""
    round_texts: list = field(default_factory=list)     # text spoken in each model round


TextCB = Callable[[str], Awaitable[None]]
EventCB = Callable[[dict], Awaitable[None]]


def _close_dangling_tool_use(s: Session, reason: str) -> None:
    """Every tool_use needs a tool_result next. Patch history if a turn was cut short."""
    if not s.history or s.history[-1]["role"] != "assistant":
        return
    calls = [b for b in s.history[-1]["content"] if getattr(b, "type", None) == "tool_use"]
    if calls:
        s.history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": reason, "is_error": True}
            for c in calls]})


async def answer(s: Session, user_text: str, on_text: TextCB, on_event: EventCB,
                 model: str = MODEL, effort: str = EFFORT, max_rounds: int = MAX_ROUNDS,
                 thinking: str = THINKING, variant: str | None = None,
                 cache_history: bool = CACHE_HISTORY, deep_dive: bool = DEEP_DIVE,
                 tools: list | None = None, max_tokens: int = MAX_TOKENS,
                 system_prompt: str | None = None, beta: dict | None = None) -> TurnStats:
    stats = TurnStats()
    t0 = time.monotonic()
    # Stable prefix: tools + instructions + overview + map, cached explicitly.
    if tools is None:
        tools = TOOLS + ([DEEP_DIVE_TOOL] if deep_dive else [])
    system = [{"type": "text", "text": system_prompt or build_stable_prompt(s, variant, deep_dive),
               "cache_control": {"type": "ephemeral"}}]
    focus = render_focus(s.focus)
    if cache_history:
        # Focus rides in this turn's user message, so the system prompt never changes and
        # the conversation so far stays a cacheable prefix across rounds and turns.
        content = [{"type": "text", "text": focus}, {"type": "text", "text": user_text}] if focus else user_text
        s.history.append({"role": "user", "content": content})
        cache_opts = {"cache_control": {"type": "ephemeral"}}   # auto-breakpoint on the last block
    else:
        # Stage 1-7 layout: focus as a per-turn system block (history never cached).
        s.history.append({"role": "user", "content": user_text})
        if focus:
            system.append({"type": "text", "text": focus})
        cache_opts = {}

    round_start = False

    async def emit_text(t: str):
        nonlocal round_start
        if round_start and stats.text and not stats.text[-1].isspace() and t and not t[0].isspace():
            t = " " + t      # text from separate rounds would otherwise run together
        round_start = False
        if stats.round_texts:
            stats.round_texts[-1] += t
        if stats.first_text_s is None:
            stats.first_text_s = time.monotonic() - t0
        stats.text += t
        await on_text(t)

    try:
        json_retries = 0
        last_read: dict | None = None
        while stats.rounds < max_rounds:
            stats.rounds += 1
            round_start = True
            stats.round_texts.append("")
            try:
                api = client().beta.messages if beta else client().messages
                async with api.stream(
                    model=model, max_tokens=max_tokens, system=system,
                    tools=tools, messages=s.history, **(beta or {}),
                    output_config={"effort": effort},
                    thinking={"type": "disabled"} if thinking == "off" else {"type": "adaptive"},
                    **cache_opts,
                ) as stream:
                    async for event in stream:
                        if event.type == "text":
                            await emit_text(event.text)
                    msg = await stream.get_final_message()
                json_retries = 0
            except ValueError:
                # Eager tool streaming: the SDK couldn't parse a tool input at all,
                # and raised before the block completed. Re-issue the turn (bounded).
                json_retries += 1
                if json_retries > MAX_JSON_RETRIES:
                    raise
                stats.rounds -= 1
                continue

            u = msg.usage
            stats.input_tokens += u.input_tokens
            stats.cache_read += u.cache_read_input_tokens or 0
            stats.cache_write += u.cache_creation_input_tokens or 0
            stats.output_tokens += u.output_tokens
            stats.stop_reasons.append(msg.stop_reason)

            s.history.append({"role": "assistant", "content": msg.content})

            if msg.stop_reason == "refusal":
                _close_dangling_tool_use(s, "refused")
                await emit_text(" Sorry, I can't help with that one.")
                break

            calls = [b for b in msg.content if b.type == "tool_use"]
            if not calls:
                break

            if msg.stop_reason == "max_tokens":
                # The last tool input was likely cut off mid-generation: don't run anything.
                _close_dangling_tool_use(
                    s, "error: output hit max_tokens before this call finished; call fewer tools at once")
                continue

            # show_code only ever returns "ok", so if the model already gave its answer
            # alongside its highlights, another model call would just add a round trip.
            answered = any(b.type == "text" and b.text.strip() for b in msg.content)
            highlight_only = answered and all(c.name == "show_code" for c in calls)

            for c in calls:
                stats.tool_calls.append((c.name, c.input))
                await on_event({"type": "tool_call", "name": c.name, "input": c.input,
                                "ts": time.monotonic()})

            async def run_call(c):
                if c.name == "deep_dive":
                    q = c.input.get("question") if isinstance(c.input, dict) else None
                    if not isinstance(q, str) or not q.strip():
                        return "error: missing required field 'question'", True
                    return await run_deep_dive(s, q, on_event), False
                return await run_tool(s, c.name, c.input)

            results = await asyncio.gather(*(run_call(c) for c in calls))
            s.history.append({"role": "user", "content": [   # ALL results in ONE message
                {"type": "tool_result", "tool_use_id": c.id, "content": content,
                 **({"is_error": True} if is_err else {})}
                for c, (content, is_err) in zip(calls, results)]})
            show_failed = False
            for c, (_, is_err) in zip(calls, results):
                if c.name == "read_file" and not is_err and isinstance(c.input, dict):
                    last_read = c.input
                if c.name == "show_code":
                    if is_err:
                        show_failed = True
                    else:
                        stats.shown.append(c.input.get("path"))
            # Stop early only if the highlights worked; a failed one gets a round to fix it.
            if highlight_only and not show_failed:
                break
        else:
            await emit_text(" That one's taking a while. Want me to keep digging?")

        # Safety net: the answer ended with nothing new on screen. Show (1) the last window
        # this turn read, else (2) the definition of a code name the answer mentions. These
        # aren't model tool calls, so they stay out of history and out of stats.shown (the
        # eval's strict grounding metric).
        if stats.shown:
            stats.highlight = "model"
        elif s.focus.get("shown_at", 0) > t0:
            stats.highlight = "model"      # shown by a helper this turn (deep_dive)
        else:
            target, source = None, None
            if last_read and last_read.get("path"):
                start = int(last_read.get("start") or 1)
                end = min(int(last_read.get("end") or start + AUTO_SHOW_LINES - 1), start + AUTO_SHOW_LINES - 1)
                target, source = (last_read["path"], start, end), "read-fallback"
            else:
                target = await locate_mentioned_code(s, stats.text)
                source = "text-fallback" if target else None
            if target:
                path, start, end = target
                _, is_err = await run_tool(s, "show_code", {"path": path, "start": start, "end": end})
                if not is_err:
                    stats.auto_shown, stats.highlight = path, source
    except asyncio.CancelledError:
        _close_dangling_tool_use(s, "interrupted by the user")
        raise
    finally:
        stats.total_s = time.monotonic() - t0
    return stats
