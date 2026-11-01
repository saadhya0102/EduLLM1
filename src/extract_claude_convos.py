"""Extract Claude Code conversation transcripts for this repo into Markdown.

Claude Code (and Claude Desktop) store every session as a JSONL file under
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` where the project folder
name is the working directory with every non-alphanumeric character replaced by
a dash (e.g. ``C:\\Users\\hasto\\EduLLM1`` -> ``C--Users-hasto-EduLLM1``).

Each JSONL line is one event. This script keeps the parts that make up the
readable conversation (user prompts, assistant thinking / text / tool calls, and
tool results) and renders them to Markdown, one file per session.

Usage:
  python -m src.extract_claude_convos                # this repo -> ./claude_conversations
  python -m src.extract_claude_convos --out docs/chats
  python -m src.extract_claude_convos --project-dir "C--Users-hasto-EduLLM1"
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"


def encode_cwd(path: Path) -> str:
    """Reproduce Claude Code's project-folder encoding for a working directory."""
    return re.sub(r"[^a-zA-Z0-9]", "-", str(path))


def find_project_dir(repo: Path, override: str | None) -> Path:
    """Locate the ~/.claude/projects folder that holds this repo's sessions."""
    if override:
        cand = Path(override)
        return cand if cand.is_absolute() else PROJECTS_DIR / override
    target = encode_cwd(repo)
    exact = PROJECTS_DIR / target
    if exact.is_dir():
        return exact
    # Fall back to a case-insensitive match (drive letter casing varies).
    if PROJECTS_DIR.is_dir():
        for d in PROJECTS_DIR.iterdir():
            if d.is_dir() and d.name.lower() == target.lower():
                return d
    return exact  # return the expected path so the caller can report it


def fmt_ts(iso: str | None) -> str:
    """Render an ISO timestamp as 'YYYY-MM-DD HH:MM:SS UTC' (best effort)."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return iso


def text_of(content) -> str:
    """Flatten a message ``content`` (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in ("text", "thinking"):
                    parts.append(b.get("text") or b.get("thinking") or "")
                elif "text" in b:
                    parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return ""


def blockquote(s: str) -> str:
    return "\n".join("> " + line if line else ">" for line in s.strip().splitlines())


def fenced(s: str, lang: str = "") -> str:
    fence = "```"
    while fence in s:  # widen the fence if the payload contains backticks
        fence += "`"
    return f"{fence}{lang}\n{s}\n{fence}"


def render_tool_result(content, max_chars: int) -> str:
    body = text_of(content).strip()
    if not body:
        # Some results carry structured (non-text) payloads.
        body = json.dumps(content, ensure_ascii=False, indent=2)
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n... [truncated {len(body) - max_chars} chars]"
    return body


def render_session(rows: list[dict], max_chars: int) -> str:
    meta = next((r for r in rows if r.get("type") == "user"), {})
    title = ""
    for r in rows:
        if r.get("type") == "ai-title" and r.get("aiTitle"):
            title = r["aiTitle"]
    session_id = meta.get("sessionId", "unknown")
    cwd = meta.get("cwd", "")
    branch = meta.get("gitBranch", "")
    version = meta.get("version", "")
    started = fmt_ts(meta.get("timestamp"))

    n_user = sum(
        1
        for r in rows
        if r.get("type") == "user"
        and isinstance(r.get("message", {}).get("content"), str)
    )
    n_asst = sum(1 for r in rows if r.get("type") == "assistant")

    out: list[str] = []
    out.append(f"# {title or 'Claude Code conversation'}")
    out.append("")
    out.append(f"- **Session:** `{session_id}`")
    if cwd:
        out.append(f"- **Project:** `{cwd}`")
    if branch:
        out.append(f"- **Git branch:** `{branch}`")
    if started:
        out.append(f"- **Started:** {started}")
    if version:
        out.append(f"- **Claude Code version:** {version}")
    out.append(f"- **Turns:** {n_user} user / {n_asst} assistant records")
    out.append("")

    # Claude Code persists one record per streamed content block, so a single
    # logical assistant turn spans many records (and the tool results it gets
    # back arrive as interleaved "user" records). Group all of that under one
    # "Assistant" header per turn, opened lazily so empty blocks add no noise.
    in_assistant = False

    def open_assistant(ts: str, model: str) -> None:
        nonlocal in_assistant
        if not in_assistant:
            out.append("---\n")
            header = f"### Assistant &middot; {ts}"
            if model:
                header += f"  _(model: {model})_"
            out.append(header)
            out.append("")
            in_assistant = True

    for r in rows:
        t = r.get("type")
        msg = r.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        ts = fmt_ts(r.get("timestamp"))

        is_tool_result = isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )

        if t == "user" and not is_tool_result:
            body = text_of(content).strip()
            if body:
                in_assistant = False
                out.append("---\n")
                out.append(f"### User &middot; {ts}")
                out.append("")
                out.append(body)
                out.append("")

        elif t == "user" and is_tool_result:
            # Tool output fed back to the model belongs to the assistant turn.
            open_assistant(ts, "")
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append("**Tool result:**")
                    out.append("")
                    out.append(fenced(render_tool_result(b.get("content"), max_chars)))
                    out.append("")

        elif t == "assistant" and isinstance(content, list):
            model = msg.get("model", "")
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "thinking":
                    think = (b.get("thinking") or "").strip()
                    if think:
                        open_assistant(ts, model)
                        out.append("<details><summary>Thinking</summary>\n")
                        out.append(blockquote(think))
                        out.append("\n</details>")
                        out.append("")
                elif bt == "text":
                    txt = (b.get("text") or "").strip()
                    if txt:
                        open_assistant(ts, model)
                        out.append(txt)
                        out.append("")
                elif bt == "tool_use":
                    open_assistant(ts, model)
                    name = b.get("name", "?")
                    out.append(f"**Tool call &rarr; `{name}`**")
                    out.append("")
                    out.append(fenced(json.dumps(b.get("input", {}), ensure_ascii=False, indent=2), "json"))
                    out.append("")

    return "\n".join(out).rstrip() + "\n"


def slugify(s: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return slug[:60] or fallback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="Repo working directory (default: cwd).")
    ap.add_argument("--project-dir", default=None, help="Override the ~/.claude/projects folder name or path.")
    ap.add_argument("--out", default="claude_conversations", help="Output directory for Markdown.")
    ap.add_argument("--max-result-chars", type=int, default=8000, help="Truncate tool results longer than this.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    proj = find_project_dir(repo, args.project_dir)
    if not proj.is_dir():
        raise SystemExit(f"No Claude project folder found at: {proj}\n(Looked under {PROJECTS_DIR})")

    sessions = sorted(proj.glob("*.jsonl"))
    subagents = sorted(proj.glob("*/subagents/*.jsonl"))
    if not sessions and not subagents:
        raise SystemExit(f"No session transcripts (*.jsonl) found in {proj}")

    out_dir = repo / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[str] = ["# Claude conversations", "", f"Extracted from `{proj}`", ""]
    written = []
    for f in sessions + subagents:
        try:
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skip {f.name}: {e}")
            continue
        if not rows:
            continue
        md = render_session(rows, args.max_result_chars)
        title_line = md.splitlines()[0].lstrip("# ").strip()
        date = ""
        for r in rows:
            if r.get("timestamp"):
                date = fmt_ts(r["timestamp"])[:10]
                break
        stem = f.stem
        name = f"{date + '_' if date else ''}{slugify(title_line, stem)}_{stem[:8]}.md"
        (out_dir / name).write_text(md, encoding="utf-8")
        written.append((name, title_line, date, len(rows)))
        print(f"  wrote {name}  ({len(rows)} events)")

    for name, title_line, date, n in written:
        index.append(f"- [{title_line}]({name}) &mdash; {date}, {n} events")
    (out_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\nDone. {len(written)} conversation(s) -> {out_dir}")


if __name__ == "__main__":
    main()
