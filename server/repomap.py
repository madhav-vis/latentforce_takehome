"""Aider-style repo map: tree-sitter tags + PageRank over the file graph.

Deterministic, no model involved:
1. Parse each file and run the grammar's tags query -> definitions and references.
2. Edge user -> definer for every identifier one file references and another defines.
3. PageRank the file graph; files many others depend on rise to the top.
4. Render the top files with their key definition lines (and line numbers, so the
   agent can read_file a window directly), within a token budget.

Tags queries come from tree-sitter-language-pack (the grammars' own tags.scm files),
so there's nothing vendored. Name matching is approximate (two `save`s in different
files both get edges); that's fine for a map that only guides where to look.
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx
import tree_sitter as ts
import tree_sitter_language_pack as tslp

log = logging.getLogger(__name__)

# Some tags queries build on another language's (upstream they "inherit").
INHERITS = {"typescript": ["javascript"], "tsx": ["javascript"]}
# Definitions worth showing in the map. Constants/variables still create edges.
SHOWN_DEFS = {"class", "function", "method", "interface", "module", "type",
              "enum", "struct", "trait", "macro", "implementation", "object"}
MAX_DEFS_PER_FILE = 25
MAX_LINE_CHARS = 110
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Tag:
    kind: str     # "def" | "ref"
    sub: str      # class, function, call, ...
    name: str
    line: int     # 1-based line of the name
    text: str     # source line of the definition (defs only)
    start: int = 0   # first line of the whole definition (defs only)
    end: int = 0     # last line of the whole definition (defs only)


def language_for(path: str) -> str | None:
    lang = tslp.detect_language_from_path(path)
    return lang if lang and _query_source(lang) else None


@lru_cache(maxsize=None)
def _query_source(lang: str) -> str | None:
    try:
        own = tslp.get_tags_query(lang)
    except Exception:
        return None
    if not own:
        return None
    parts = [tslp.get_tags_query(base) or "" for base in INHERITS.get(lang, [])]
    return "\n".join(parts + [own])


@lru_cache(maxsize=None)
def _compiled(lang: str) -> tuple[ts.Parser, ts.Query] | None:
    try:
        return tslp.get_parser(lang), ts.Query(tslp.get_language(lang), _query_source(lang))
    except Exception as e:      # missing grammar, query that doesn't compile, download failure
        log.warning("no tags for %s: %s", lang, e)
        return None


def prefetch_languages(langs: set[str]) -> None:
    """Grammars download on first use; fetch everything this repo needs in one go."""
    missing = sorted(langs - set(tslp.downloaded_languages()))
    if missing:
        try:
            tslp.download(missing)
        except Exception as e:
            log.warning("grammar download failed for %s: %s", missing, e)


def extract_tags(root: str, path: str, lang: str) -> list[Tag]:
    compiled = _compiled(lang)
    if not compiled:
        return []
    parser, query = compiled
    try:
        src = (Path(root) / path).read_bytes()
    except OSError:
        return []
    tree = parser.parse(src)
    lines = src.decode(errors="replace").splitlines()
    tags = []
    for _, caps in ts.QueryCursor(query).matches(tree.root_node):
        kinds = [k for k in caps if k != "name"]
        if not kinds:
            continue
        kind, _, sub = kinds[0].partition(".")
        # JS/TS object-literal callbacks (`onSuccess: () => {}`) aren't real definitions.
        if kind == "definition" and any(n.type == "pair" for n in caps[kinds[0]]):
            continue
        if kind == "definition":
            kind = "def"
        elif kind == "reference":
            kind = "ref"
        else:
            continue
        whole = caps[kinds[0]][0]
        for node in caps.get("name", []):
            name = node.text.decode(errors="replace")
            line = node.start_point[0] + 1
            text = lines[line - 1].rstrip() if kind == "def" and line <= len(lines) else ""
            tags.append(Tag(kind, sub, name, line, text,
                            whole.start_point[0] + 1, whole.end_point[0] + 1))
    return tags


def build_repo_map(root: str, files: list[str], token_budget: int = 3000) -> tuple[str, dict]:
    """Returns (rendered map, file graph {file: {file it uses: weight}})."""
    langs = {f: language_for(f) for f in files}
    prefetch_languages({l for l in langs.values() if l})

    defines: dict[str, set[str]] = defaultdict(set)       # name -> files defining it
    references: dict[str, Counter] = defaultdict(Counter)  # name -> {file: count}
    def_lines: dict[str, list[Tag]] = defaultdict(list)
    for path in files:
        if not langs[path]:
            continue
        for tag in extract_tags(root, path, langs[path]):
            if tag.kind == "def":
                defines[tag.name].add(path)
                def_lines[path].append(tag)
            else:
                references[tag.name][path] += 1

    g = nx.DiGraph()
    g.add_nodes_from(def_lines)
    for name, users in references.items():
        definers = defines.get(name, ())
        if not definers:
            continue
        # Aider's damping: common names defined everywhere say little; private names less.
        mul = 0.1 if len(definers) > 5 else 1.0
        if name.startswith("_"):
            mul *= 0.1
        for user, count in users.items():
            for definer in definers:
                if user != definer:
                    w = g.get_edge_data(user, definer, {}).get("weight", 0)
                    g.add_edge(user, definer, weight=w + mul * math.sqrt(count))

    try:
        rank = nx.pagerank(g, weight="weight") if g.number_of_edges() else {}
    except nx.PowerIterationFailedConvergence:
        rank = {}
    ordered = sorted(files, key=lambda f: (-rank.get(f, 0), -len(def_lines.get(f, ())), f))
    # file -> {file it uses: edge weight}; weights let focus hints rank strong links first
    graph = {f: {d: round(g[f][d]["weight"], 2) for d in g.successors(f)} for f in g.nodes if g.out_degree(f)}

    # How many other files reference each name: used to pick a file's key definitions.
    ref_files = {name: len(users) for name, users in references.items()}
    return render_map(ordered, def_lines, ref_files, token_budget), graph


def render_map(ordered: list[str], def_lines: dict[str, list[Tag]],
               ref_files: dict[str, int], token_budget: int) -> str:
    budget = token_budget * CHARS_PER_TOKEN
    out, used = [], 0
    for path in ordered:
        shown = [t for t in def_lines.get(path, ()) if t.sub in SHOWN_DEFS and t.text.strip()]
        if not shown:
            continue
        if len(shown) > MAX_DEFS_PER_FILE:
            keep = sorted(shown, key=lambda t: (-ref_files.get(t.name, 0), t.line))[:MAX_DEFS_PER_FILE]
            shown = sorted(keep, key=lambda t: t.line)
        seen, block = set(), [path]
        for t in shown:
            if t.line in seen:
                continue
            seen.add(t.line)
            text = t.text if len(t.text) <= MAX_LINE_CHARS else t.text[:MAX_LINE_CHARS] + "…"
            block.append(f"{t.line:>5} {text}")
        chunk = "\n".join(block) + "\n"
        if used + len(chunk) > budget:
            # Skip this file but keep trying smaller ones; stop when even a bare path won't fit.
            if used + len(path) + 1 > budget:
                break
            continue
        out.append(chunk)
        used += len(chunk)
    return "".join(out).rstrip()


def render_file_list(files: list[str], token_budget: int = 3000) -> str:
    """Fallback when no file has parseable tags (docs-only repos, unsupported languages)."""
    budget, out, used = token_budget * CHARS_PER_TOKEN, [], 0
    for f in files:
        if used + len(f) + 1 > budget:
            out.append(f"[{len(files) - len(out)} more files]")
            break
        out.append(f)
        used += len(f) + 1
    return "\n".join(out)


# --- step 6: which symbol is on screen ------------------------------------------------

CONTAINERS = {"class", "interface", "module", "struct", "trait", "implementation", "object", "enum"}
RANGE_SLACK = 3     # a shown range may include a decorator or a line or two past the end


def enclosing_symbol(root: str, path: str, start: int, end: int) -> str | None:
    """Qualified name of the innermost definition containing lines start..end, e.g.
    'AuthService.refresh_token'. Deterministic: the same tags the repo map uses."""
    lang = language_for(path)
    if not lang:
        return None
    defs = [t for t in extract_tags(root, path, lang) if t.kind == "def" and t.end >= t.start > 0]
    around = [t for t in defs if t.start <= start + RANGE_SLACK and t.end >= end - RANGE_SLACK]
    if not around:
        return None
    around.sort(key=lambda t: t.end - t.start, reverse=True)     # outermost first
    inner = around[-1]
    chain = [t.name for t in around[:-1] if t.sub in CONTAINERS] + [inner.name]
    return ".".join(dict.fromkeys(chain))


def definition_end(root: str, path: str, line: int) -> int | None:
    """Last line of the definition starting (or named) at `line`, from the tags."""
    lang = language_for(path)
    if not lang:
        return None
    for t in extract_tags(root, path, lang):
        if t.kind == "def" and line in (t.start, t.line):
            return t.end
    return None
