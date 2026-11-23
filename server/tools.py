"""Read-only tools confined to the clone. Repo code is never executed."""

import asyncio
import re
import time
from pathlib import Path

from .repomap import definition_end, enclosing_symbol
from .session import Session, emit
from .setup import NOISE_DIRS, run

MAX_RESULT_CHARS = 8000
MAX_GREP_MATCHES = 50
MAX_READ_LINES = 200
MAX_DIR_ENTRIES = 200
SHOW_CONTEXT = 5
MAX_SHOW_LINES = 120        # cap what one highlight pushes to the screen


class ToolError(Exception):
    """Reported back to the model as an is_error tool_result."""


def safe(root: str, rel: str) -> Path:
    base = Path(root).resolve()
    p = (base / rel.lstrip("/")).resolve()   # resolve() follows symlinks, so escapes are caught
    if not p.is_relative_to(base):
        raise ToolError("path outside repo")
    return p


def _cap(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + "\n[truncated]"


def _read_text(p: Path) -> list[str]:
    if not p.is_file():
        raise ToolError(f"not a file: {p.name}")
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        raise ToolError("binary file")
    return data.decode(errors="replace").splitlines()


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(lines))


# --- tools -----------------------------------------------------------------

async def grep(s: Session, pattern: str, path_glob: str | None = None) -> str:
    args = ["rg", "-n", "--no-heading", "--color", "never", "--max-count", "5",
            "--max-columns", "300", "--max-columns-preview"]
    for d in sorted(NOISE_DIRS):
        args += ["-g", f"!{d}/"]
    if path_glob:
        args += ["-g", path_glob]
    args += ["-e", pattern, "--", "."]
    try:
        code, out, err = await run(args, cwd=s.repo_path, timeout=10)
    except asyncio.TimeoutError:
        raise ToolError("grep timed out after 10s; narrow the pattern or path_glob")
    if code == 1:
        return "no matches"
    if code != 0:
        raise ToolError(err.strip() or "grep failed")
    lines = [l.removeprefix("./") for l in out.splitlines()]
    more = len(lines) - MAX_GREP_MATCHES
    body = "\n".join(lines[:MAX_GREP_MATCHES])
    if more > 0:
        body += f"\n[{more} more matches not shown]"
    return _cap(body)


async def read_file(s: Session, path: str, start: int = 1, end: int | None = None) -> str:
    lines = _read_text(safe(s.repo_path, path))
    start = max(1, start)
    end = min(len(lines), end if end is not None else start + MAX_READ_LINES - 1)
    end = min(end, start + MAX_READ_LINES - 1)
    if start > len(lines):
        return f"[{path} has only {len(lines)} lines]"
    _push_recent(s, path.removeprefix("./"))
    header = f"{path} lines {start}-{end} of {len(lines)}\n"
    return _cap(header + _numbered(lines[start - 1:end], start))


async def list_dir(s: Session, path: str = ".") -> str:
    p = safe(s.repo_path, path)
    if not p.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = []
    for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
        if child.name in NOISE_DIRS:
            continue
        entries.append(child.name + ("/" if child.is_dir() else ""))
    more = len(entries) - MAX_DIR_ENTRIES
    body = "\n".join(entries[:MAX_DIR_ENTRIES]) or "(empty)"
    if more > 0:
        body += f"\n[{more} more entries not shown]"
    return _cap(body)


async def show_code(s: Session, path: str, start: int, end: int) -> str:
    lines = _read_text(safe(s.repo_path, path))
    start = min(max(1, start), max(1, len(lines)))
    end = min(len(lines), max(start, end), start + MAX_SHOW_LINES - 1)
    lo, hi = max(1, start - SHOW_CONTEXT), min(len(lines), end + SHOW_CONTEXT)
    path = path.removeprefix("./")
    symbol = await asyncio.to_thread(enclosing_symbol, s.repo_path, path, start, end)
    # Raw lines: the page and CLI draw their own line numbers and highlight [start, end].
    await emit(s, "show_code", path=path, code=lines[lo - 1:hi], first_line=lo,
               lines=[start, end], total_lines=len(lines), symbol=symbol)
    update_focus(s, path, start, end, symbol)
    # The model only sees this string, not the code. It sometimes uses show_code to "look"
    # at something it just promised to check, so say what to do next either way.
    return ("Shown on the user's screen (you have not seen these lines; use read_file if you "
            "need them). If your spoken answer is already complete, end your turn with no "
            "more text. Otherwise keep going.")


# --- step 6: focus ---------------------------------------------------------------
# What "that" refers to. Updated by tool calls, never by a model.

FOCUS_NEIGHBOURS = 5


def update_focus(s: Session, path: str, start: int, end: int, symbol: str | None):
    """show_code is the strongest signal: it's what the user is looking at."""
    used_by = {f: uses[path] for f, uses in s.graph.items() if path in uses}
    uses = s.graph.get(path, {})
    strongest = lambda d: sorted(d, key=lambda f: (-d[f], f))[:FOCUS_NEIGHBOURS]
    s.focus.update(
        file=path, lines=[start, end], symbol=symbol,
        used_by=strongest(used_by),                       # likely next reads
        uses=strongest(uses),
        shown_at=time.monotonic(),                        # lets a turn tell if anything was shown
    )
    _push_recent(s, path)


def _push_recent(s: Session, path: str):
    recent = s.focus.setdefault("recent_files", [])
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    del recent[FOCUS_NEIGHBOURS:]


def render_focus(focus: dict) -> str:
    """The per-turn CURRENT FOCUS block (after the cache breakpoint)."""
    if not focus.get("file") and not focus.get("recent_files"):
        return ""
    out = ['CURRENT FOCUS (what "this", "that", "it" and "there" most likely refer to)']
    if focus.get("file"):
        a, b = focus.get("lines") or [None, None]
        where = f"{focus['file']} lines {a}-{b}" if a else focus["file"]
        sym = f", inside {focus['symbol']}" if focus.get("symbol") else ""
        out.append(f"- On screen now: {where}{sym}")
    if focus.get("used_by"):
        out.append(f"- Files that use it: {', '.join(focus['used_by'])}")
    if focus.get("uses"):
        out.append(f"- Files it uses: {', '.join(focus['uses'])}")
    others = [f for f in focus.get("recent_files", []) if f != focus.get("file")]
    if others:
        out.append(f"- Recently read: {', '.join(others)}")
    return "\n".join(out)


# --- schemas ---------------------------------------------------------------
# eager_input_streaming: inputs stream as generated, so we validate them ourselves (validate_input).

TOOLS = [
    {
        "name": "grep",
        "description": "Search the repo with ripgrep (regex). Returns file:line: text, max 5 matches per file, 50 total.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Rust-regex pattern, e.g. 'def login' or 'class \\w+Service'"},
                "path_glob": {"type": "string", "description": "Optional glob to limit files, e.g. '*.py' or 'src/auth/**'"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read numbered lines from a file (repo-relative path). Max 200 lines per call; prefer ~60-line windows.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "description": "1-based first line (default 1)"},
                "end": {"type": "integer", "description": "Last line, inclusive"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List a directory (repo-relative; '.' for the root). Folders end with '/'.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "show_code",
        "description": "Show these lines on the user's screen. Call it for the code you are talking about.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["path", "start", "end"],
        },
    },
]

HANDLERS = {"grep": grep, "read_file": read_file, "list_dir": list_dir, "show_code": show_code}
_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
_PY_TYPES = {"string": str, "integer": int}


def validate_input(name: str, inp) -> dict:
    """Check a (possibly partial) streamed input against its schema. Raises ToolError."""
    schema = _SCHEMAS.get(name)
    if schema is None:
        raise ToolError(f"unknown tool {name}")
    if not isinstance(inp, dict):
        raise ToolError("input must be an object")
    for key in schema["required"]:
        if key not in inp:
            raise ToolError(f"missing required field '{key}'")
    clean = {}
    for key, val in inp.items():
        prop = schema["properties"].get(key)
        if prop is None:
            raise ToolError(f"unexpected field '{key}'")
        want = _PY_TYPES[prop["type"]]
        if want is int and isinstance(val, str) and val.strip().lstrip("-").isdigit():
            val = int(val)
        if not isinstance(val, want) or isinstance(val, bool):
            raise ToolError(f"field '{key}' must be {prop['type']}")
        clean[key] = val
    return clean


async def run_tool(s: Session, name: str, inp) -> tuple[str, bool]:
    """Returns (content, is_error)."""
    try:
        args = validate_input(name, inp)
        return await HANDLERS[name](s, **args), False
    except ToolError as e:
        return f"error: {e}", True
    except OSError as e:
        return f"error: {e.strerror or e}", True



# --- fallback highlight: find the code an answer talks about ----------------------------

_BACKTICKED = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
_CODE_NAME = re.compile(r"\b[A-Za-z_]+(?:_[A-Za-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_DEF_KEYWORDS = r"(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:def|class|function|func|fn|interface|type|struct|trait|enum|const|let|var)"
MAX_CANDIDATES = 6


def code_names(text: str) -> list[str]:
    """Identifiers an answer mentions: backticked names first, then snake_case/camelCase/CamelCase."""
    names = [m.split(".")[-1] for m in _BACKTICKED.findall(text)] + _CODE_NAME.findall(text)
    return list(dict.fromkeys(n for n in names if len(n) >= 4))[:MAX_CANDIDATES]


async def locate_mentioned_code(s: Session, text: str) -> tuple[str, int, int] | None:
    """(path, start, end) of the first definition the answer names, or None."""
    for name in code_names(text):
        pattern = rf"^\s*{_DEF_KEYWORDS}\s+{re.escape(name)}\b"
        try:
            code, out, _ = await run(["rg", "-n", "--no-heading", "--color", "never", "--max-count", "1",
                                      *[a for d in sorted(NOISE_DIRS) for a in ("-g", f"!{d}/")],
                                      "-e", pattern, "--", "."], cwd=s.repo_path, timeout=5)
        except asyncio.TimeoutError:
            continue
        hits = [l.removeprefix("./").split(":", 2) for l in out.splitlines() if l.count(":") >= 2]
        # prefer real source over tests and examples
        hits.sort(key=lambda h: any(part in h[0] for part in ("test", "example", "docs/")))
        for path, line, _ in hits[:1]:
            line = int(line)
            end = await asyncio.to_thread(definition_end, s.repo_path, path, line)
            return path, line, min(end or line + 30, line + 59)
    return None
