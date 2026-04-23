"""Agentic Architectures 4: Planning (CLI application).

Implements the Planning architecture and contrasts it against the ReAct
(Reason + Act) baseline from notebook 03.

The Planning agent decomposes a complex goal into a complete, ordered list
of sub-tasks BEFORE executing anything. A `Planner` LLM produces the plan,
an `Executor` runs each step against a web-search tool, and a final
`Synthesizer` LLM stitches the collected observations into a single answer.
The ReAct agent, by contrast, reasons one step at a time in a loop.

Both agents share the same Gemini LLM and a Tavily web-search tool, plumbed
through LangGraph. The CLI can run either agent on a query, run both side by
side, optionally evaluate the resulting traces with an LLM-as-a-judge, and
print the agent workflow graphs.

Run examples:
    # Run the planning agent on the canonical multi-step demo query
    python 04_planning_CLI.py

    # Run a custom query with the planning agent only
    python 04_planning_CLI.py "Compare the GDP of France, Germany and Italy."

    # Head-to-head comparison of the ReAct vs. planning agent
    python 04_planning_CLI.py --mode both "Combined population of EU big three vs. USA?"

    # Compare both, then run the LLM-as-a-judge evaluation on each trace
    python 04_planning_CLI.py --mode both --evaluate

    # Interactive REPL with the planning agent
    python 04_planning_CLI.py --interactive

    # Just print the workflow graphs of both agents (no LLM calls run)
    python 04_planning_CLI.py --show-graphs

Required environment variables (loaded from a local .env file if present):
    GOOGLE_API_KEY    - Google Gemini API key
    TAVILY_API_KEY    - Tavily search API key
    LANGCHAIN_API_KEY - optional, enables LangSmith tracing with --trace
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Annotated, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from utils import extract_text, print_message

console = Console()

DEFAULT_PLAN_CENTRIC_QUERY = (
    "Find the population of the capital cities of France, Germany, and Italy. "
    "Then calculate their combined total. Finally, compare that combined total "
    "to the population of the United States, and say which is larger."
)


# --- Environment Setup ---
def load_environment(enable_tracing: bool) -> List[str]:
    """Load .env and return a list of missing required keys."""
    load_dotenv(override=True)

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enable_tracing else "false"
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        "Agentic Architecture - Planning (Google Gemini)",
    )

    required = ["GOOGLE_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    return missing


# --- Graph State ---
class AgentState(TypedDict):
    """Conversation history shared across the ReAct graph."""

    messages: Annotated[list[AnyMessage], add_messages]


class PlanningState(TypedDict):
    """State for the planning agent graph."""

    user_request: str
    plan: Optional[List[str]]
    intermediate_steps: List[ToolMessage]
    final_answer: Optional[str]


# --- Structured-Output Schemas ---
class Plan(BaseModel):
    """A plan of tool calls to execute to answer the user's query."""

    steps: List[str] = Field(
        description="A list of tool calls that, when executed, will answer the query."
    )


class ProcessEvaluation(BaseModel):
    """Schema for evaluating an agent's problem-solving process."""

    task_completion_score: int = Field(
        description="Score 1-10 on whether the agent successfully completed the task."
    )
    process_efficiency_score: int = Field(
        description=(
            "Score 1-10 on the efficiency and directness of the agent's process. "
            "Higher means a more logical and less roundabout path."
        )
    )
    justification: str = Field(description="A brief justification for the scores.")


# --- Shared Building Blocks ---
def build_search_tool(max_results: int) -> TavilySearch:
    """Create the Tavily search tool used by both agents."""
    return TavilySearch(max_results=max_results)


def build_llm(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Create the shared Gemini LLM."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=20)


def make_web_search_tool(tavily: TavilySearch):
    """Wrap the raw Tavily tool with a logging shim and a stable tool name."""

    @tool
    def web_search(query: str) -> str:
        """Performs a web search using Tavily and returns the results as a string."""
        console.print(f"[dim]--- TOOL: Searching for '{query}'... ---[/dim]")
        return tavily.invoke(query)

    return web_search


# --- ReAct Agent ---
def build_react_agent_app(llm: ChatGoogleGenerativeAI, web_search_tool):
    """Compile the ReAct agent with a `think -> act -> observe` loop."""
    tools = [web_search_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(
        content=(
            "You are a helpful research assistant. You must call one and only "
            "one tool at a time. Do not call multiple tools in a single turn. "
            "After receiving the result from a tool, you will decide on the "
            "next step."
        )
    )

    def react_agent_node(state: AgentState):
        """Reason about the current state and decide on the next action."""
        console.print("[dim]--- REACT AGENT: Thinking... ---[/dim]")
        messages = [system_prompt] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", react_agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


# --- Planning Agent ---
_TOOL_CALL_RE = re.compile(r"(\w+)\((?:\"|\')(.*?)(?:\"|\')\)")


def _parse_tool_call(step: str) -> tuple[str, str]:
    """Extract `(tool_name, query)` from a planner step like `web_search('foo')`.

    Falls back to treating the entire string as a `web_search` query when the
    planner produces a less structured step.
    """
    match = _TOOL_CALL_RE.search(step)
    if not match:
        return "web_search", step
    return match.group(1), match.group(2)


def build_planning_agent_app(llm: ChatGoogleGenerativeAI, tavily: TavilySearch):
    """Compile the Planner -> Executor (loop) -> Synthesizer graph."""
    planner_llm = llm.with_structured_output(Plan)

    def planner_node(state: PlanningState):
        """Generate a plan of action to answer the user's request."""
        console.print("[dim]--- PLANNER: Decomposing task... ---[/dim]")
        prompt = (
            "You are an expert planner. Your job is to create a step-by-step "
            "plan to answer the user's request. Each step in the plan must be "
            "a single call to the `web_search` tool.\n\n"
            "**Instructions:**\n"
            "1. Analyze the user's request.\n"
            "2. Break it down into a sequence of simple, logical search queries.\n"
            "3. Format the output as a list of strings, where each string is a "
            "single valid tool call.\n\n"
            "**Example:**\n"
            'Request: "What is the capital of France and what is its population?"\n'
            "Correct Plan Output:\n"
            "[\n"
            "    \"web_search('capital of France')\",\n"
            "    \"web_search('population of Paris')\"\n"
            "]\n\n"
            f"**User's Request:**\n{state['user_request']}\n"
        )
        plan_result = planner_llm.invoke(prompt)
        console.print(
            Panel(
                JSON.from_data(plan_result.steps),
                title="PLANNER - Generated Plan",
                border_style="blue",
            )
        )
        return {"plan": plan_result.steps}

    def executor_node(state: PlanningState):
        """Execute the next step in the plan."""
        console.print("[dim]--- EXECUTOR: Running next step... ---[/dim]")
        plan = state["plan"] or []
        next_step = plan[0]
        tool_name, query = _parse_tool_call(next_step)
        console.print(
            f"[dim]--- EXECUTOR: Calling tool '{tool_name}' with query "
            f"'{query}' ---[/dim]"
        )

        result = tavily.invoke(query)
        tool_message = ToolMessage(
            content=str(result),
            name=tool_name,
            tool_call_id=f"manual-{abs(hash(query))}",
        )
        return {
            "plan": plan[1:],
            "intermediate_steps": state["intermediate_steps"] + [tool_message],
        }

    def synthesizer_node(state: PlanningState):
        """Synthesize the final answer from the intermediate steps."""
        console.print("[dim]--- SYNTHESIZER: Generating final answer... ---[/dim]")
        context = "\n".join(
            f"Tool {msg.name} returned: {msg.content}"
            for msg in state["intermediate_steps"]
        )
        prompt = (
            "You are an expert synthesizer. Based on the user's request and "
            "the collected data, provide a comprehensive final answer.\n\n"
            f"Request: {state['user_request']}\n"
            f"Collected Data:\n{context}\n"
        )
        final_answer = llm.invoke(prompt).content
        if isinstance(final_answer, list):
            final_answer = "\n".join(
                part if isinstance(part, str) else part.get("text", str(part))
                for part in final_answer
            )
        return {"final_answer": final_answer}

    def planning_router(state: PlanningState) -> str:
        """Route to the executor while plan steps remain, else synthesize."""
        if not state["plan"]:
            console.print(
                "[dim]--- ROUTER: Plan complete. Moving to synthesizer. ---[/dim]"
            )
            return "synthesize"
        console.print(
            "[dim]--- ROUTER: Plan has more steps. Continuing execution. ---[/dim]"
        )
        return "execute"

    builder = StateGraph(PlanningState)
    builder.add_node("plan", planner_node)
    builder.add_node("execute", executor_node)
    builder.add_node("synthesize", synthesizer_node)
    builder.set_entry_point("plan")
    builder.add_conditional_edges(
        "plan", planning_router, {"execute": "execute", "synthesize": "synthesize"}
    )
    builder.add_conditional_edges(
        "execute", planning_router, {"execute": "execute", "synthesize": "synthesize"}
    )
    builder.add_edge("synthesize", END)
    return builder.compile()


# --- Run a Query ---
def run_react_agent(app, user_query: str, label: str, label_style: str) -> dict:
    """Stream the ReAct agent over a single query and return its final state."""
    console.print(
        Rule(
            f"[bold {label_style}]{label} - Query:[/bold {label_style}] {user_query}",
            style=label_style,
        )
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
        Rule(
            f"[bold {label_style}]{label} - Workflow complete[/bold {label_style}]",
            style=label_style,
        )
    )
    return final_state or {"messages": []}


def run_planning_agent(app, user_query: str, label: str, label_style: str) -> dict:
    """Invoke the planning agent and render each intermediate step."""
    console.print(
        Rule(
            f"[bold {label_style}]{label} - Query:[/bold {label_style}] {user_query}",
            style=label_style,
        )
    )

    initial_input: PlanningState = {
        "user_request": user_query,
        "plan": None,
        "intermediate_steps": [],
        "final_answer": None,
    }
    final_state: Optional[dict] = None
    seen_ids: set[int] = set()

    for chunk in app.stream(initial_input, stream_mode="values"):
        final_state = chunk
        for msg in chunk.get("intermediate_steps") or []:
            if id(msg) in seen_ids:
                continue
            seen_ids.add(id(msg))
            print_message(msg)

    console.print(
        Rule(
            f"[bold {label_style}]{label} - Workflow complete[/bold {label_style}]",
            style=label_style,
        )
    )
    return final_state or {"user_request": user_query, "intermediate_steps": []}


# --- Final Answer Rendering ---
def _find_final_answer_message(messages: list) -> Optional[Any]:
    """Return the last AI message that carries real text content."""
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


def render_react_final_answer(final_state: dict, label: str, border_style: str) -> None:
    """Print the final assistant answer from a ReAct run as Markdown."""
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


def render_planning_final_answer(
    final_state: dict, label: str, border_style: str
) -> None:
    """Print the final synthesized answer from a planning run as Markdown."""
    answer = final_state.get("final_answer")
    console.print()
    if not answer:
        console.print(
            Panel(
                "[yellow]No final answer was synthesised.[/yellow]",
                title=f"{label} - Final Answer",
                border_style=border_style,
            )
        )
        return

    if isinstance(answer, list):
        answer = "\n".join(
            part if isinstance(part, str) else part.get("text", str(part))
            for part in answer
        )

    console.print(
        Panel(Markdown(str(answer)), title=f"{label} - Final Answer", border_style=border_style)
    )


# --- Evaluation ---
def _react_trace(final_state: dict) -> str:
    return "\n".join(
        f"{m.type}: {extract_text(getattr(m, 'content', ''))} "
        f"{getattr(m, 'tool_calls', '') or ''}"
        for m in final_state.get("messages", [])
    )


def _planning_trace(final_state: dict) -> str:
    plan = final_state.get("plan") or []
    steps = final_state.get("intermediate_steps") or []
    rendered_steps = "\n".join(
        f"tool {m.name}: {extract_text(getattr(m, 'content', ''))}"
        for m in steps
    )
    return (
        f"Plan (remaining): {plan}\n"
        f"Executed steps:\n{rendered_steps}\n"
        f"Final answer: {final_state.get('final_answer', '')}"
    )


def evaluate_trace(
    llm: ChatGoogleGenerativeAI,
    query: str,
    final_state: dict,
    kind: str,
) -> ProcessEvaluation:
    """Run the LLM-as-a-judge evaluation on the conversation trace."""
    judge_llm = llm.with_structured_output(ProcessEvaluation)
    trace = _react_trace(final_state) if kind == "react" else _planning_trace(final_state)
    prompt = (
        "You are an expert judge of AI agents. Evaluate the agent's process for "
        "solving the task on a scale of 1-10. Focus on whether the process was "
        "logical and efficient.\n\n"
        f"**User's Task:** {query}\n"
        f"**Full Agent Trace:**\n```\n{trace}\n```\n"
    )
    return judge_llm.invoke(prompt)


def render_evaluation(label: str, evaluation: ProcessEvaluation) -> None:
    """Pretty-print a single evaluation result."""
    body = (
        f"[bold]Task completion:[/bold]      {evaluation.task_completion_score}/10\n"
        f"[bold]Process efficiency:[/bold]   {evaluation.process_efficiency_score}/10\n\n"
        f"[bold]Justification:[/bold] {evaluation.justification}"
    )
    console.print()
    console.print(
        Panel(body, title=f"LLM-as-a-Judge - {label}", border_style="magenta")
    )


# --- Graph Visualisation ---
def _render_one_graph(label: str, app, border_style: str) -> None:
    """Render a single compiled LangGraph app to the console."""
    console.print(
        Rule(
            f"[bold {border_style}]{label} - Workflow Graph[/bold {border_style}]",
            style=border_style,
        )
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
        console.print(mermaid_src)
    except Exception as exc:
        console.print(f"[red]Mermaid rendering failed: {exc}[/red]")


def render_agent_graphs(react_app, planning_app) -> None:
    """Render both the ReAct and planning agent graphs."""
    _render_one_graph("REACT AGENT", react_app, border_style="red")
    console.print()
    _render_one_graph("PLANNING AGENT", planning_app, border_style="green")


# --- High-level Orchestration ---
def run_mode(
    mode: str,
    query: str,
    react_app,
    planning_app,
    llm: ChatGoogleGenerativeAI,
    evaluate: bool,
) -> None:
    """Dispatch to the requested agent(s) and optionally evaluate the trace(s)."""
    runs: list[tuple[str, str, str, dict]] = []

    if mode in ("react", "both"):
        state = run_react_agent(react_app, query, label="REACT", label_style="red")
        render_react_final_answer(state, label="ReAct Agent", border_style="red")
        runs.append(("ReAct Agent", "react", "red", state))

    if mode in ("plan", "both"):
        state = run_planning_agent(
            planning_app, query, label="PLANNING", label_style="green"
        )
        render_planning_final_answer(state, label="Planning Agent", border_style="green")
        runs.append(("Planning Agent", "planning", "green", state))

    if evaluate:
        for label, kind, _border, state in runs:
            try:
                evaluation = evaluate_trace(llm, query, state, kind=kind)
                render_evaluation(label, evaluation)
            except Exception as exc:
                console.print(f"[red]Evaluation failed for {label}: {exc}[/red]")


# --- CLI ---
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="04_planning_CLI",
        description=(
            "Run a Planning agent (Gemini + Tavily search) - and optionally a "
            "ReAct agent for comparison - over a single query or in an "
            "interactive REPL."
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help=(
            "The user query to send to the agent. Omit to read from stdin, "
            "use --interactive, or fall back to the built-in plan-centric demo query."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("react", "plan", "both"),
        default="plan",
        help=(
            "Which agent to run: 'react' (looping), 'plan' (decompose-then-execute), "
            "or 'both' for a head-to-head comparison (default: plan)."
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
            "Print the workflow graph of BOTH agents (ReAct and Planning) to "
            "the console as ASCII (if `grandalf` is installed) and Mermaid source. "
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

    missing = load_environment(enable_tracing=args.trace)
    if missing:
        console.print(
            f"[red]Missing some required environment variables: "
            f"{', '.join(missing)}.[/red]\n"
            "Create a .env file in the project root with the required keys."
        )
        return 1

    llm = build_llm(model=args.model, temperature=args.temperature)
    tavily = build_search_tool(max_results=args.max_results)
    web_search_tool = make_web_search_tool(tavily)
    react_app = build_react_agent_app(llm, web_search_tool)
    planning_app = build_planning_agent_app(llm, tavily)

    if args.show_graphs:
        render_agent_graphs(react_app, planning_app)

    initial_query = _resolve_initial_query(args)
    if not initial_query and not args.interactive and not args.show_graphs:
        console.print(
            f"[yellow]No query provided; using built-in demo query:[/yellow]\n"
            f"  {DEFAULT_PLAN_CENTRIC_QUERY}\n"
        )
        initial_query = DEFAULT_PLAN_CENTRIC_QUERY

    if initial_query:
        run_mode(
            mode=args.mode,
            query=initial_query,
            react_app=react_app,
            planning_app=planning_app,
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
                    react_app=react_app,
                    planning_app=planning_app,
                    llm=llm,
                    evaluate=args.evaluate,
                )
            except Exception as exc:
                console.print(f"[red]Error while running query: {exc}[/red]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
