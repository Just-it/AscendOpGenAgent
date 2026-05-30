"""JSON helpers shared by subprocess result reading and history.jsonl
loading. Single source for both — duplicates in phase_machine and
task_config were collapsed here."""
from __future__ import annotations

import json
import os
from typing import List, Optional


def _read_whole_file(path: str) -> str:
    """Loop os.read until EOF — `open().read()` short-reads on large
    history.jsonl, silently dropping the tail."""
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        os.close(fd)


def load_jsonl(path: str) -> List[dict]:
    """Every JSON object in a JSONL file. Missing file → []. Blank
    and malformed lines are skipped."""
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    for line in _read_whole_file(path).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def parse_last_json_line(text: str) -> Optional[dict]:
    """Last `{...}` line in `text`, parsed. None if no line is valid
    JSON. Non-JSON lines after the result don't cause false negatives."""
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
