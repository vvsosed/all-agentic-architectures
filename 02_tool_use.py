"""Agentic Architectures 2: Tool Use (CLI application).

A tool-using agent that combines a Google Gemini LLM with the Tavily web search
tool via LangGraph. The agent autonomously decides when to call the search
tool, observes the results, and synthesises a final, grounded answer.

Run examples:
    # One-shot query
    python 02_tool_use.py "Is there any evidence that time machines exist?"

    # Interactive mode (REPL); type 'exit' or Ctrl-D to quit
    python 02_tool_use.py --interactive

    # Run with LLM-as-a-judge evaluation of the trace
    python 02_tool_use.py "Latest SpaceX launch?" --evaluate

Required environment variables (loaded from a local .env file if present):
    GOOGLE_API_KEY   - Google Gemini API key
    TAVILY_API_KEY   - Tavily search API key
    LANGCHAIN_API_KEY (optional) - enables LangSmith tracing
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown, Json
from rich.panel import Panel
from rich.rule import Rule

console = Console()


# --- Environment Setup ---
def load_environment(enable_tracing: bool) -> List[str]:
    """Load .env and return a list of missing required keys."""
    load_dotenv(override=True)

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enable_tracing else "false"
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        "Agentic Architecture - Tool Use (Google Gemini)",
    )

    required = ["GOOGLE_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    return missing


# --- Graph State ---
class AgentState(TypedDict):
    """Conversation history shared across the graph."""

    messages: Annotated[list[AnyMessage], add_messages]


# --- Evaluation Schema ---
class ToolUseEvaluation(BaseModel):
    """Schema for evaluating the agent's tool use and final answer."""

    tool_selection_score: int = Field(
        description="Score 1-5 on whether the agent chose the correct tool for the task."
    )
    tool_input_score: int = Field(
        description="Score 1-5 on how well-formed and relevant the input to the tool was."
    )
    synthesis_quality_score: int = Field(
        description="Score 1-5 on how well the agent integrated the tool's output into its final answer."
    )
    justification: str = Field(description="A brief justification for the scores.")


# --- Agent Construction ---
def build_search_tool(max_results: int) -> TavilySearchResults:
    """Create the Tavily search tool with an agent-friendly description."""
    search_tool = TavilySearchResults(max_results=max_results)
    search_tool.name = "web_search"
    search_tool.description = (
        "A tool that can be used to search the internet for up-to-date information "
        "on any topic, including news, events, and current affairs."
    )
    return search_tool


def build_agent_app(model: str, temperature: float, max_results: int):
    """Compile the tool-using LangGraph agent and return (app, llm)."""
    search_tool = build_search_tool(max_results=max_results)
    tools = [search_tool]

    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=10)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        """Call the LLM to decide whether to answer or call a tool."""
        console.print("[dim]--- AGENT: Thinking... ---[/dim]")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def router_function(state: AgentState) -> str:
        """Route to the tool node if a tool call is requested, else finish."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            console.print("[dim]--- ROUTER: Decision is to call a tool. ---[/dim]")
            return "call_tool"
        console.print("[dim]--- ROUTER: Decision is to finish. ---[/dim]")
        return "__end__"

    tool_node = ToolNode(tools)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("call_tool", tool_node)
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", router_function)
    graph_builder.add_edge("call_tool", "agent")

    return graph_builder.compile(), llm


# --- Pretty Printing ---
def _message_title(message) -> str:
    """Return a human-readable title for a LangChain message."""
    cls = type(message).__name__
    return f"{cls} ({getattr(message, 'type', '?')})"


def print_message(message) -> None:
    """Render a LangChain message to the console."""
    title = _message_title(message)
    body_parts: List[str] = []

    content = getattr(message, "content", "")
    if content:
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        body_parts.append(str(content))

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        body_parts.append("[bold]Tool calls:[/bold]")
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            body_parts.append(f"  - {name}({args})")

    body = "\n".join(body_parts) if body_parts else "[dim](empty)[/dim]"

    if tool_calls:
        console.print(Panel.fit(Json(tool_calls), title="Tool Calls", border_style="cyan"))
    else:
        console.print(Panel.fit(Markdown(body), title=title, border_style="cyan"))


# --- Run a Query ---
def run_query(app, user_query: str) -> dict:
    """Stream the agent over a single user query and return the final state."""
    console.print(
        Rule(f"[bold cyan]Query:[/bold cyan] {user_query}", style="cyan")
    )

    initial_input = {"messages": [("user", user_query)]}
    final_state: Optional[dict] = None
    seen_ids: set[int] = set()

    for chunk in app.stream(initial_input, stream_mode="values"):
        final_state = chunk
        last = chunk["messages"][-1]
        if id(last) in seen_ids:
            continue
        seen_ids.add(id(last))
        print_message(last)

    console.print(Rule("[bold green]Workflow complete[/bold green]", style="green"))
    return final_state or {"messages": []}


def render_final_answer(final_state: dict) -> None:
    """Print the final assistant answer as Markdown."""
    if not final_state.get("messages"):
        return
    last = final_state["messages"][-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not content:
        return
    console.print()
    console.print(Panel(Markdown(str(content)), title="Final Answer", border_style="green"))


# --- Evaluation ---
def evaluate_trace(llm, final_state: dict) -> ToolUseEvaluation:
    """Run the LLM-as-a-judge evaluation on the conversation trace."""
    judge_llm = llm.with_structured_output(ToolUseEvaluation)
    trace = "\n".join(
        f"{m.type}: {getattr(m, 'content', '') or ''} {getattr(m, 'tool_calls', '')}"
        for m in final_state.get("messages", [])
    )
    prompt = (
        "You are an expert judge of AI agents. Evaluate the following conversation "
        "trace based on the agent's tool use on a scale of 1-5. Provide a brief "
        "justification.\n\nConversation Trace:\n```\n"
        f"{trace}\n```\n"
    )
    return judge_llm.invoke(prompt)


def render_evaluation(evaluation: ToolUseEvaluation) -> None:
    """Pretty-print the evaluation result."""
    console.print()
    body = (
        f"[bold]Tool selection:[/bold]   {evaluation.tool_selection_score}/5\n"
        f"[bold]Tool input:[/bold]       {evaluation.tool_input_score}/5\n"
        f"[bold]Synthesis quality:[/bold] {evaluation.synthesis_quality_score}/5\n\n"
        f"[bold]Justification:[/bold] {evaluation.justification}"
    )
    console.print(Panel(body, title="LLM-as-a-Judge Evaluation", border_style="magenta"))


# --- CLI ---
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="02_tool_use",
        description=(
            "Run a Tool-Use agent (Gemini + Tavily search) over a single query "
            "or in an interactive REPL."
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="The user query to send to the agent. Omit to read from stdin or use --interactive.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive REPL after handling any positional query.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Google Gemini model name (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum web search results returned by the tool (default: 3).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the LLM-as-a-judge evaluation on the conversation trace.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable LangSmith tracing (requires LANGCHAIN_API_KEY).",
    )
    return parser.parse_args(argv)


def _resolve_initial_query(args: argparse.Namespace) -> Optional[str]:
    """Combine positional args (or stdin) into a single query string."""
    if args.query:
        return " ".join(args.query).strip()
    if not args.interactive and not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        return data or None
    return None


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    missing = load_environment(enable_tracing=args.trace)
    if missing:
        console.print(
            f"[red]Missing required environment variables: {', '.join(missing)}.[/red]\n"
            "Create a .env file in the project root with the required keys."
        )
        return 1

    app, llm = build_agent_app(
        model=args.model,
        temperature=args.temperature,
        max_results=args.max_results,
    )

    initial_query = _resolve_initial_query(args)

    if not initial_query and not args.interactive:
        console.print(
            "[yellow]No query provided. Pass one as an argument, pipe via stdin, "
            "or use --interactive.[/yellow]"
        )
        return 2

    if initial_query:
        final_state = run_query(app, initial_query)
        render_final_answer(final_state)
        if args.evaluate:
            evaluation = evaluate_trace(llm, final_state)
            render_evaluation(evaluation)

    if args.interactive:
        console.print(
            "\n[bold]Interactive mode.[/bold] Type your query and press Enter. "
            "Type 'exit' or 'quit' (or send EOF) to leave.\n"
        )
        while True:
            try:
                user_query = console.input("[bold cyan]› [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit"}:
                break
            try:
                final_state = run_query(app, user_query)
                render_final_answer(final_state)
                if args.evaluate:
                    evaluation = evaluate_trace(llm, final_state)
                    render_evaluation(evaluation)
            except Exception as exc:
                console.print(f"[red]Error while running query: {exc}[/red]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
