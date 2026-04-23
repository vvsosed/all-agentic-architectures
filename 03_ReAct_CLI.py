"""Agentic Architectures 3: ReAct (CLI application).

Implements the ReAct (Reason + Act) architecture and contrasts it against a
"basic" single-shot tool-using agent. The basic agent gets one chance to call
a tool and then must answer; the ReAct agent loops `think -> act -> observe`
until it can answer.

Both agents share the same Gemini LLM and a Tavily web-search tool, plumbed
through LangGraph. The CLI can run either agent on a query, run both side by
side, and optionally evaluate the resulting traces with an LLM-as-a-judge.

Run examples:
    # Run the ReAct agent on the canonical multi-step demo question
    python 03_ReAct_CLI.py

    # Run a custom query with the ReAct agent only
    python 03_ReAct_CLI.py "Who founded the company that built the Falcon 9?"

    # Head-to-head comparison of the basic vs. ReAct agent
    python 03_ReAct_CLI.py --mode both "Who is the CEO of the company that made 'Dune'?"

    # Compare both, then run the LLM-as-a-judge evaluation on each trace
    python 03_ReAct_CLI.py --mode both --evaluate

    # Interactive REPL with the ReAct agent
    python 03_ReAct_CLI.py --interactive

    # Just print the workflow graphs of both agents (no LLM calls run)
    python 03_ReAct_CLI.py --show-graphs

    # Print graphs and then run a query with the ReAct agent
    python 03_ReAct_CLI.py --show-graphs "Latest SpaceX launch?"

Required environment variables (loaded from a local .env file if present):
    GOOGLE_API_KEY   - Google Gemini API key
    TAVILY_API_KEY   - Tavily search API key
    LANGCHAIN_API_KEY (optional) - enables LangSmith tracing with --trace
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Annotated, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from utils import extract_text, print_message

console = Console()

DEFAULT_MULTI_STEP_QUERY = (
    "Who is the current CEO of the company that created the sci-fi movie 'Dune', "
    "and what was the budget for that company's most recent film?"
)


# --- Environment Setup ---
def load_environment(enable_tracing: bool) -> List[str]:
    """Load .env and return a list of missing required keys."""
    load_dotenv(override=True)

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enable_tracing else "false"
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        "Agentic Architecture - ReAct (Google Gemini)",
    )

    required = ["GOOGLE_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    return missing


# --- Graph State ---
class AgentState(TypedDict):
    """Conversation history shared across the graph."""

    messages: Annotated[list[AnyMessage], add_messages]


# --- Evaluation Schema ---
class TaskEvaluation(BaseModel):
    """Schema for evaluating an agent's ability to complete a task."""

    task_completion_score: int = Field(
        description="Score 1-10 on whether the agent successfully completed all parts of the user's request."
    )
    reasoning_quality_score: int = Field(
        description="Score 1-10 on the logical flow and reasoning process demonstrated by the agent."
    )
    justification: str = Field(description="A brief justification for the scores.")


# --- Agent Construction ---
def build_search_tool(max_results: int) -> TavilySearchResults:
    """Create the Tavily search tool used by both agents."""
    search_tool = TavilySearchResults(max_results=max_results, name="web_search")
    search_tool.description = (
        "A tool that can be used to search the internet for up-to-date information "
        "on any topic, including news, events, and current affairs."
    )
    return search_tool


def build_llm(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Create the shared Gemini LLM."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=20)


def build_basic_agent_app(llm: ChatGoogleGenerativeAI, search_tool: TavilySearchResults):
    """Compile the basic single-shot tool-using agent.

    The agent gets exactly one chance to call a tool, after which it is forced
    to synthesise a textual answer from whatever the tool returned. There is
    no reasoning loop, so multi-hop questions still fail - just with a real
    natural-language answer instead of raw tool output.

    Graph:
        agent --tool_calls-- > tools -- > synthesize -- > END
            \\-- no tool call ---------------------------- > END
    """
    tools = [search_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "You are a helpful assistant. You have access to a web search tool. "
        "Answer the user's question based on the tool's results. You must "
        "provide a final answer after one tool call."
    )

    def basic_agent_node(state: AgentState):
        """Single LLM call, biased to answer after a single tool invocation."""
        console.print("[dim]--- BASIC AGENT: Thinking... ---[/dim]")
        messages = [("system", system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def basic_synthesize_node(state: AgentState):
        """Force a final textual answer from the tool results.

        The LLM is invoked WITHOUT tools bound here, so it cannot request
        another tool call - it must produce its best answer from the single
        batch of observations it already has.
        """
        console.print("[dim]--- BASIC AGENT: Synthesising final answer... ---[/dim]")
        synthesis_prompt = (
            "You have already used the web search tool exactly once. "
            "You MUST now produce a final natural-language answer to the "
            "user's question using ONLY the tool results above. "
            "Do not request any more tools. If the information is "
            "insufficient, say so explicitly and explain what is missing."
        )
        messages = (
            [("system", system_prompt)]
            + list(state["messages"])
            + [("system", synthesis_prompt)]
        )
        response = llm.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", basic_agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("synthesize", basic_synthesize_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


def build_react_agent_app(llm: ChatGoogleGenerativeAI, search_tool: TavilySearchResults):
    """Compile the ReAct agent with a `think -> act -> observe` loop."""
    tools = [search_tool]
    llm_with_tools = llm.bind_tools(tools)

    def react_agent_node(state: AgentState):
        """Reason about the current state and decide on the next action."""
        console.print("[dim]--- REACT AGENT: Thinking... ---[/dim]")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def react_router(state: AgentState) -> str:
        """Route to the tool node if a tool call is requested, else finish."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            console.print("[dim]--- ROUTER: Decision is to call a tool. ---[/dim]")
            return "tools"
        console.print("[dim]--- ROUTER: Decision is to finish. ---[/dim]")
        return "__end__"

    builder = StateGraph(AgentState)
    builder.add_node("agent", react_agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", react_router, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


# --- Run a Query ---
def run_agent(app, user_query: str, label: str, label_style: str) -> dict:
    """Stream the given agent over a single query and return its final state."""
    console.print(
        Rule(f"[bold {label_style}]{label} - Query:[/bold {label_style}] {user_query}",
             style=label_style)
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

    console.print(
        Rule(f"[bold {label_style}]{label} - Workflow complete[/bold {label_style}]",
             style=label_style)
    )
    return final_state or {"messages": []}


def _find_final_answer_message(messages: list) -> Optional[Any]:
    """Return the last AI message that carries real text content.

    Walks backwards over the conversation, skipping `ToolMessage`s and any AI
    messages that only contain tool-call requests with no textual body. This
    makes the renderer robust against graphs whose final state ends on a tool
    result rather than on a synthesised assistant answer.
    """
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        if getattr(message, "tool_calls", None) and not extract_text(
            getattr(message, "content", "")
        ).strip():
            continue
        if extract_text(getattr(message, "content", "")).strip():
            return message
    return None


def render_final_answer(final_state: dict, label: str, border_style: str) -> None:
    """Print the final assistant answer from a run as Markdown."""
    messages = final_state.get("messages") or []
    if not messages:
        return

    answer_msg = _find_final_answer_message(messages)
    if answer_msg is None:
        last = messages[-1]
        last_kind = type(last).__name__
        console.print()
        console.print(
            Panel(
                f"[yellow]No synthesised assistant answer was produced.[/yellow]\n"
                f"Last message in the trace was a [bold]{last_kind}[/bold]; "
                "see the streamed trace above for raw tool output.",
                title=f"{label} - Final Answer",
                border_style=border_style,
            )
        )
        return

    content = extract_text(getattr(answer_msg, "content", ""))
    console.print()
    console.print(
        Panel(Markdown(content), title=f"{label} - Final Answer", border_style=border_style)
    )


# --- Evaluation ---
def evaluate_trace(llm: ChatGoogleGenerativeAI, query: str, final_state: dict) -> TaskEvaluation:
    """Run the LLM-as-a-judge evaluation on the conversation trace."""
    judge_llm = llm.with_structured_output(TaskEvaluation)
    trace = "\n".join(
        f"{m.type}: {extract_text(getattr(m, 'content', ''))} "
        f"{getattr(m, 'tool_calls', '') or ''}"
        for m in final_state.get("messages", [])
    )
    prompt = (
        "You are an expert judge of AI agents. Evaluate the following agent's "
        "performance on the given task on a scale of 1-10. A score of 10 means "
        "the task was completed perfectly. A score of 1 means complete failure.\n\n"
        f"**User's Task:**\n{query}\n\n"
        f"**Full Agent Conversation Trace:**\n```\n{trace}\n```\n"
    )
    return judge_llm.invoke(prompt)


def render_evaluation(label: str, evaluation: TaskEvaluation) -> None:
    """Pretty-print a single evaluation result."""
    body = (
        f"[bold]Task completion:[/bold]    {evaluation.task_completion_score}/10\n"
        f"[bold]Reasoning quality:[/bold]  {evaluation.reasoning_quality_score}/10\n\n"
        f"[bold]Justification:[/bold] {evaluation.justification}"
    )
    console.print()
    console.print(
        Panel(body, title=f"LLM-as-a-Judge - {label}", border_style="magenta")
    )


# --- Graph Visualisation ---
def _render_one_graph(label: str, app, border_style: str) -> None:
    """Render a single compiled LangGraph app to the console.

    Always prints the Mermaid source (zero extra dependencies, copy-pasteable
    into any Mermaid renderer). Additionally tries an ASCII rendering, which
    requires the optional `grandalf` package - if it is not installed, the
    Mermaid block is shown alone with a short note.
    """
    console.print(
        Rule(f"[bold {border_style}]{label} - Workflow Graph[/bold {border_style}]",
             style=border_style)
    )
    graph = app.get_graph()

    try:
        ascii_art = graph.draw_ascii()
        console.print(
            Panel(
                Text(ascii_art, no_wrap=True),
                title=f"{label} - ASCII",
                border_style=border_style,
            )
        )
    except ImportError:
        console.print(
            "[dim]ASCII rendering unavailable (install `grandalf` to enable: "
            "`pip install grandalf`).[/dim]"
        )
    except Exception as exc:
        console.print(f"[dim]ASCII rendering failed: {exc}[/dim]")

    try:
        mermaid_src = graph.draw_mermaid()
        console.print(
            Panel(
                Syntax(mermaid_src, "mermaid", theme="ansi_dark", word_wrap=True),
                title=f"{label} - Mermaid",
                border_style=border_style,
            )
        )
    except Exception as exc:
        console.print(f"[red]Mermaid rendering failed: {exc}[/red]")


def render_agent_graphs(basic_app, react_app) -> None:
    """Render both the basic and ReAct agent graphs side by side."""
    _render_one_graph("BASIC AGENT", basic_app, border_style="red")
    console.print()
    _render_one_graph("REACT AGENT", react_app, border_style="green")


# --- High-level Orchestration ---
def run_mode(
    mode: str,
    query: str,
    basic_app,
    react_app,
    llm: ChatGoogleGenerativeAI,
    evaluate: bool,
) -> None:
    """Dispatch to the requested agent(s) and optionally evaluate the trace(s)."""
    runs: list[tuple[str, str, dict]] = []

    if mode in ("basic", "both"):
        state = run_agent(basic_app, query, label="BASIC", label_style="red")
        render_final_answer(state, label="Basic Agent", border_style="red")
        runs.append(("Basic Agent", "red", state))

    if mode in ("react", "both"):
        state = run_agent(react_app, query, label="REACT", label_style="green")
        render_final_answer(state, label="ReAct Agent", border_style="green")
        runs.append(("ReAct Agent", "green", state))

    if evaluate:
        for label, _border, state in runs:
            try:
                evaluation = evaluate_trace(llm, query, state)
                render_evaluation(label, evaluation)
            except Exception as exc:
                console.print(f"[red]Evaluation failed for {label}: {exc}[/red]")


# --- CLI ---
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="03_ReAct_CLI",
        description=(
            "Run a ReAct agent (Gemini + Tavily search) - and optionally a "
            "basic single-shot agent for comparison - over a single query "
            "or in an interactive REPL."
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help=(
            "The user query to send to the agent. Omit to read from stdin, "
            "use --interactive, or fall back to the built-in multi-step demo query."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("basic", "react", "both"),
        default="react",
        help=(
            "Which agent to run: 'basic' (single-shot), 'react' (looping), "
            "or 'both' for a head-to-head comparison (default: react)."
        ),
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
        default=2,
        help="Maximum web search results returned by the tool (default: 2).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the LLM-as-a-judge evaluation on the conversation trace(s).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable LangSmith tracing (requires LANGCHAIN_API_KEY).",
    )
    parser.add_argument(
        "-g",
        "--show-graphs",
        action="store_true",
        help=(
            "Print the workflow graph of BOTH agents (basic and ReAct) to the "
            "console as ASCII (if `grandalf` is installed) and Mermaid source. "
            "Can be combined with a query; if used alone, the demo query is "
            "NOT auto-run."
        ),
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

    is_missing = load_environment(enable_tracing=args.trace)
    if is_missing:
        console.print(
            f"[red]Missing some required environment variables: {', '.join(is_missing)}.[/red]\n"
            "Create a .env file in the project root with the required keys."
        )
        return 1

    llm = build_llm(model=args.model, temperature=args.temperature)
    search_tool = build_search_tool(max_results=args.max_results)
    basic_app = build_basic_agent_app(llm, search_tool)
    react_app = build_react_agent_app(llm, search_tool)

    if args.show_graphs:
        render_agent_graphs(basic_app, react_app)

    initial_query = _resolve_initial_query(args)
    if not initial_query and not args.interactive and not args.show_graphs:
        console.print(
            f"[yellow]No query provided; using built-in demo query:[/yellow]\n"
            f"  {DEFAULT_MULTI_STEP_QUERY}\n"
        )
        initial_query = DEFAULT_MULTI_STEP_QUERY

    if initial_query:
        run_mode(
            mode=args.mode,
            query=initial_query,
            basic_app=basic_app,
            react_app=react_app,
            llm=llm,
            evaluate=args.evaluate,
        )

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
                run_mode(
                    mode=args.mode,
                    query=user_query,
                    basic_app=basic_app,
                    react_app=react_app,
                    llm=llm,
                    evaluate=args.evaluate,
                )
            except Exception as exc:
                console.print(f"[red]Error while running query: {exc}[/red]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
