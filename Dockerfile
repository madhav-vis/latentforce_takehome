FROM python:3.12-slim-bookworm

# Bookworm has glibc 2.36; the tree-sitter-language-pack wheels need 2.34+.
# git clones the repos, ripgrep backs the grep tool, libglib is needed by opencv-headless (pulled in by pipecat's WebRTC extra).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# tree-sitter grammars download on first use. Fetch the common ones now so the
# first repo map doesn't wait on downloads; rarer languages still download on demand.
RUN python -c "import tree_sitter_language_pack as t; t.download(['python','javascript','typescript','tsx','go','rust','java','kotlin','c','cpp','csharp','ruby','php','swift','scala','lua','elixir','dart'])"

COPY server ./server
COPY web ./web

ENV HOST=0.0.0.0 PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "server.main"]
