#!/usr/bin/env python3
"""将 Claude Code session jsonl 渲染为可读 markdown。

用法:
    python render_session.py <session.jsonl> <session.md>
"""
import json
import sys
from pathlib import Path


def _extract_text(message):
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                try:
                    inp_s = json.dumps(inp, ensure_ascii=False)[:500]
                except Exception:
                    inp_s = str(inp)[:500]
                parts.append(f"[tool_use: {name}] {inp_s}")
            elif btype == "tool_result":
                out = block.get("content", "")
                if isinstance(out, list):
                    out = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in out
                    )
                parts.append(f"[tool_result]\n{str(out)[:2000]}")
        return "\n\n".join(p for p in parts if p)
    return ""


def render(src: Path, dst: Path):
    lines = ["# Claude Code Session\n"]
    with src.open() as f:
        for raw in f:
            try:
                d = json.loads(raw)
            except Exception:
                continue
            t = d.get("type")
            if t == "user":
                text = _extract_text(d.get("message", {}))
                if text.strip():
                    lines.append(f"## 🧑 User\n\n{text}\n")
            elif t == "assistant":
                text = _extract_text(d.get("message", {}))
                if text.strip():
                    lines.append(f"## 🤖 Assistant\n\n{text}\n")
    dst.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: render_session.py <src.jsonl> <dst.md>", file=sys.stderr)
        sys.exit(2)
    render(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Rendered: {sys.argv[2]}")
