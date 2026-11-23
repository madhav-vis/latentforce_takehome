"""Stage 1: text-only CLI. Paste a repo URL, ask questions in the terminal.

    python -m server.cli [https://github.com/owner/repo]

Ctrl-C during an answer cancels it (a stand-in for barge-in); Ctrl-C or Ctrl-D at the
prompt quits.
"""

import asyncio
import json
import os
import shutil
import signal
import sys

from dotenv import load_dotenv

DIM, CYAN, YELLOW, RED, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


class Printer:
    """Keeps streamed answer text and side events from landing on the same line."""

    def __init__(self):
        self.mid_line = False

    def stream(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        self.mid_line = not text.endswith("\n")

    def line(self, text: str):
        if self.mid_line:
            sys.stdout.write("\n")
        print(text, flush=True)
        self.mid_line = False


def fmt_event(ev: dict) -> str:
    t = ev["type"]
    if t == "progress":
        return f"{DIM}· {ev['text']}{RESET}"
    if t == "error":
        return f"{RED}✗ {ev['text']}{RESET}"
    if t == "tool_call":
        return f"{DIM}  → {ev['name']}({json.dumps(ev['input'], ensure_ascii=False)}){RESET}"
    if t == "show_code":
        a, b = ev["lines"]
        out = [f"{YELLOW}┌ {ev['path']}  lines {a}-{b}{RESET}"]
        for num, line in enumerate(ev["code"], start=ev["first_line"]):
            row = f"{num:>5}  {line}"
            mark = f"{YELLOW}│{RESET}" if a <= num <= b else f"{DIM}│{RESET}"
            out.append(f"{mark} {row if a <= num <= b else DIM + row + RESET}")
        return "\n".join(out)
    return f"{DIM}{ev}{RESET}"


async def drain_events(s, p: Printer):
    while True:
        p.line(fmt_event(await s.events.get()))


async def ask(prompt: str) -> str | None:
    try:
        return await asyncio.to_thread(input, prompt)
    except EOFError:
        return None


async def main():
    load_dotenv()
    from .agent import answer
    from .session import SESSIONS, Session
    from .setup import SetupError, setup_repo

    url = sys.argv[1] if len(sys.argv) > 1 else await ask("GitHub repo URL: ")
    if not url:
        return
    s = Session(repo_url=url.strip())
    SESSIONS[s.id] = s
    p = Printer()
    drainer = asyncio.create_task(drain_events(s, p))

    try:
        try:
            await setup_repo(s)
        except SetupError:
            await asyncio.sleep(0.05)   # let the drainer print the error event
            return
        await asyncio.sleep(0.05)
        p.line(f"{DIM}{s.repo_name} @ {s.commit[:10]}  ({s.repo_path}){RESET}")
        if s.overview:
            p.line(f"{CYAN}bot>{RESET} {s.overview}")

        async def on_text(t):
            p.stream(t)

        async def on_event(ev):
            await s.events.put(ev)

        while True:
            q = await ask(f"\n{CYAN}you>{RESET} ")
            if q is None or q.strip() in {"exit", "quit"}:
                break
            if not q.strip():
                continue
            p.stream(f"{CYAN}bot>{RESET} ")
            task = asyncio.create_task(answer(s, q.strip(), on_text, on_event), name="answer")
            try:
                st = await task
            except asyncio.CancelledError:
                p.line(f"{DIM}[interrupted]{RESET}")
                continue
            await asyncio.sleep(0.02)   # flush any trailing events
            first = f"{st.first_text_s:.2f}s" if st.first_text_s is not None else "-"
            p.line(f"{DIM}[rounds={st.rounds} tools={len(st.tool_calls)} first_text={first} "
                   f"total={st.total_s:.2f}s in={st.input_tokens} cache_read={st.cache_read} "
                   f"cache_write={st.cache_write} out={st.output_tokens} "
                   f"stop={','.join(st.stop_reasons)}]{RESET}")
    finally:
        drainer.cancel()
        if s.repo_path:
            shutil.rmtree(s.repo_path, ignore_errors=True)


def run():
    async def runner():
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()

        def on_sigint():
            # Ctrl-C: cancel the in-flight answer if there is one, otherwise quit.
            answering = [t for t in asyncio.all_tasks()
                         if t.get_name() == "answer" and not t.done()]
            for t in answering:
                t.cancel()
            if not answering:
                main_task.cancel()

        loop.add_signal_handler(signal.SIGINT, on_sigint)
        try:
            await main()
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    print()
    sys.stdout.flush()
    os._exit(0)   # don't wait on a thread still blocked in input()


if __name__ == "__main__":
    run()
