"""Shared utilities for the agentic-architecture CLI demos.

Provides content normalisation and rich-console pretty printing helpers used
across the various agent CLIs.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from rich.console import Console, Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()


def extract_text(content) -> str:
    """Extract plain text from LLM content that may be a string or a list of blocks."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


# --- Pretty Printing ---
def _message_title(message) -> str:
    """Return a human-readable title for a LangChain message."""
    cls = type(message).__name__
    return f"{cls} ({getattr(message, 'type', '?')})"


def _try_parse_json(text: str) -> Optional[Any]:
    """Return parsed JSON if `text` looks like a JSON document, else None."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return None


def _looks_like_tavily(parsed: Any) -> bool:
    """Heuristic: a non-empty list of dicts with at least title/url/content keys."""
    return (
        isinstance(parsed, list)
        and bool(parsed)
        and all(isinstance(x, dict) for x in parsed)
        and all({"title", "url", "content"}.issubset(x.keys()) for x in parsed)
    )


def _maybe_render_tool_payload(parsed: Any) -> Optional[tuple[Group, int]]:
    """Render Tavily-style payloads. Handles both raw lists and dict envelopes.

    Returns a `(rendered_group, result_count)` tuple, or `None` if the payload
    does not look like Tavily search results.
    """
    if _looks_like_tavily(parsed):
        return _render_tavily_results(parsed), len(parsed)
    if isinstance(parsed, dict):
        nested = parsed.get("results")
        if _looks_like_tavily(nested):
            return _render_tavily_results(nested), len(nested)
    return None


def _render_tavily_results(results: List[dict]) -> Group:
    """Render a list of Tavily search-result dicts as stacked rich panels."""
    panels = []
    for i, item in enumerate(results, start=1):
        title = item.get("title") or "(untitled)"
        url = item.get("url") or ""
        score = item.get("score")
        snippet = item.get("content") or ""

        header = Text()
        header.append(f"{i}. {title}\n", style="bold")
        if url:
            header.append(url, style=f"link {url} blue underline")
        if score is not None:
            try:
                header.append(f"   (score: {float(score):.4f})", style="dim")
            except (TypeError, ValueError):
                header.append(f"   (score: {score})", style="dim")

        panels.append(
            Panel(
                Group(header, Text(""), Markdown(snippet)),
                border_style="dim",
                padding=(0, 1),
            )
        )
    return Group(*panels)


def print_message(message, target_console: Optional[Console] = None) -> None:
    """Render a LangChain message (Human/AI/Tool) to the console."""
    out = target_console or console
    title = _message_title(message)
    msg_type = getattr(message, "type", "")
    raw_content = extract_text(getattr(message, "content", ""))
    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        out.print(
            Panel.fit(
                JSON.from_data(tool_calls),
                title=f"{title} - tool calls",
                border_style="cyan",
            )
        )
        return

    if msg_type == "tool":
        parsed = _try_parse_json(raw_content)
        if parsed is not None:
            rendered = _maybe_render_tool_payload(parsed)
            if rendered is not None:
                group, count = rendered
                out.print(
                    Panel(
                        group,
                        title=f"{title} - {count} result(s)",
                        border_style="yellow",
                    )
                )
                return
            out.print(
                Panel(JSON.from_data(parsed), title=title, border_style="yellow")
            )
            return

    body = raw_content if raw_content else "[dim](empty)[/dim]"
    out.print(Panel.fit(Markdown(body), title=title, border_style="cyan"))
