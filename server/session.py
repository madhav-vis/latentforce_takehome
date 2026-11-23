"""Per-session state. Everything lives in memory; there is no database."""

import asyncio
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    repo_url: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    repo_name: str = ""           # owner/repo
    repo_path: str = ""
    commit: str = ""
    overview: str = ""            # 3-sentence spoken summary (stage 2)
    repo_map: str = ""            # ranked text map, goes in the prompt (stage 2)
    graph: dict = field(default_factory=dict)   # file -> {file it uses: weight}
    focus: dict = field(default_factory=dict)   # step 6
    history: list = field(default_factory=list) # Claude messages
    cloned: asyncio.Event = field(default_factory=asyncio.Event)  # tools usable
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    error: str = ""              # setup failure, if any
    events: asyncio.Queue = field(default_factory=asyncio.Queue)


SESSIONS: dict[str, Session] = {}


async def emit(s: Session, type_: str, **data) -> None:
    await s.events.put({"type": type_, **data})
