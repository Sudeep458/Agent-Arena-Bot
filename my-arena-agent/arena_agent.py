"""
arena_agent.py — Agent Arena Multi-Turn Google ADK Agent
=========================================================

A fully autonomous multi-turn agent that navigates the Agent Arena:
  - Registers once, then loops indefinitely
  - Fetches the assigned task for the current level
  - Solves it and submits
  - On LEVEL_UP → fetches the next level's task and continues
  - On NO_TASKS or repeated failure → reports and exits cleanly
  - Prints a running scoreboard after each task attempt

Dependencies
------------
    pip install google-adk fastmcp traceloop-sdk google-genai litellm python-dotenv

Usage
-----
    uv run python agent.py
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Optional

from config import load_config

try:
    _CFG = load_config()
except Exception as e:
    print(f"Configuration error: {e}")
    raise SystemExit(1)

# ── Google ADK ────────────────────────────────────────────────────────────────
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# ── FastMCP ───────────────────────────────────────────────────────────────────
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

# ── Traceloop ─────────────────────────────────────────────────────────────────
from traceloop.sdk import Traceloop, set_association_properties
from traceloop.sdk.decorators import workflow
from traceloop.sdk.tracing import set_conversation_id

# ── OTel logging ──────────────────────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.resources import Resource

# ── Dynamic prompts ───────────────────────────────────────────────────────────
from prompts import build_task_prompt, detect_task_type

# ── LiteLLM (optional — enables OpenCode Go and other providers) ─────────────
try:
    from google.adk.models.lite_llm import LiteLlm
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MCP_ENDPOINT = "https://agent-arena.dev/mcp"

ID_TOKEN = _CFG.get("ID_TOKEN", "")

AGENT_NAME    = "sudeep-arena-adk-agent"
LINKEDIN_URL  = "www.linkedin.com/in/sudeep-devadiga-631817231"
GITHUB_URL    = "https://github.com/Sudeep458/Agent-Arena-Bot"
GEMINI_MODEL   = "gemini-3-flash-preview"
GEMINI_API_KEY = _CFG.get("GEMINI_API_KEY", "")
TRACELOOP_API_KEY = _CFG.get("TRACELOOP_API_KEY", "")

OPENCODE_GO_API_KEY = _CFG.get("OPENCODE_GO_API_KEY", "")
OPENCODE_GO_MODEL   = _CFG.get("OPENCODE_GO_MODEL", "kimi-k2.6")
OPENCODE_GO_BASE    = "https://opencode.ai/zen/go/v1"

MAX_TASKS = 20


def _active_model():
    if OPENCODE_GO_API_KEY and _LITELLM_AVAILABLE:
        return LiteLlm(
            model=f"openai/{OPENCODE_GO_MODEL}",
            api_base=OPENCODE_GO_BASE,
            api_key=OPENCODE_GO_API_KEY,
        )
    return GEMINI_MODEL


def _active_model_name() -> str:
    if OPENCODE_GO_API_KEY and _LITELLM_AVAILABLE:
        return f"opencode-go/{OPENCODE_GO_MODEL}"
    return GEMINI_MODEL


AGENT_STACK = f"Python / Google ADK / {_active_model_name()} / Traceloop"


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(tag: str, msg: str, level: str = "INFO") -> None:
    emoji = {
        "REGISTER": "📝", "FETCH": "📥", "SUBMIT": "📤",
        "SCORE": "🏆", "LEVEL": "🚀", "SKIP": "⏭️",
        "ERROR": "❌", "WARN": "⚠️", "DONE": "✅",
        "TASK": "📋", "LOOP": "🔄", "AGENT": "🤖",
        "TRACE": "📡", "RECOVER": "🔧",
    }.get(tag, "•")
    print(f"[{_ts()}] {emoji} [{tag}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Run-scoped state
# ─────────────────────────────────────────────────────────────────────────────

class RunState:
    def __init__(self) -> None:
        self.run_id       = str(uuid.uuid4())
        self.execution_id = str(uuid.uuid4())
        self.agent_id     = ""
        self.task_id      = ""
        self.conversation_id = ""

        self.current_level = 1
        self.total_score   = 0
        self.tasks_attempted = 0
        self.tasks_passed    = 0
        self.level_history: list[dict] = []

        self.current_task: Optional[dict] = None

    def record(self, level: int, task_title: str, score: int, levelled_up: bool) -> None:
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up or score >= 70:
            self.tasks_passed += 1
        if levelled_up:
            self.current_level = level + 1
        self.level_history.append({
            "level": level, "task": task_title,
            "score": score, "levelled_up": levelled_up,
        })

    def scoreboard(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  SCOREBOARD  (run {self.run_id[:8]})  model: {_active_model_name()}",
            f"{'─'*60}",
            f"  Current Level : {self.current_level}",
            f"  Total Score   : {self.total_score}",
            f"  Tasks Done    : {self.tasks_attempted}  (passed: {self.tasks_passed})",
            f"{'─'*60}",
        ]
        for entry in self.level_history:
            icon = "✅" if entry["levelled_up"] else ("🟡" if entry["score"] >= 70 else "❌")
            lines.append(
                f"  {icon} L{entry['level']}  {entry['task'][:40]:<40}  {entry['score']:>3}/100"
            )
        lines.append(f"{'─'*60}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OTel / Traceloop logging
# ─────────────────────────────────────────────────────────────────────────────

class _OtelOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        tid = getattr(record, "otelTraceID", "0")
        return tid not in ("0", "00000000000000000000000000000000", None, "")


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(h)
    return logger


agent_logger = _make_logger("arena.agent")
task_logger  = _make_logger("arena.task")


def init_tracing() -> None:
    Traceloop.init(
        app_name="arena-adk-agent",
        api_key=TRACELOOP_API_KEY or None,
        disable_batch=True,
        telemetry_enabled=False,
    )
    log_provider = LoggerProvider(resource=Resource.create({"service.name": "arena-adk-agent"}))
    exporter = ConsoleLogExporter()
    if TRACELOOP_API_KEY:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        exporter = OTLPLogExporter(
            endpoint="https://api.traceloop.com/v1/logs",
            headers={"Authorization": f"Bearer {TRACELOOP_API_KEY}", "x-traceloop-sdk-version": "traceloop-sdk"},
        )
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    for logger in (agent_logger, task_logger):
        h = LoggingHandler(logger_provider=log_provider)
        h.setLevel(logging.INFO)
        h.addFilter(_OtelOnlyFilter())
        logger.addHandler(h)
    _log("TRACE", "Traceloop initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# MCP helper
# ─────────────────────────────────────────────────────────────────────────────

async def _mcp_call(tool_name: str, arguments: dict, state: RunState) -> str:
    from fastmcp.exceptions import ToolError
    transport = StreamableHttpTransport(url=MCP_ENDPOINT)
    try:
        async with Client(transport=transport, name="arena-adk-agent") as client:
            set_association_properties({
                "execution.id": state.execution_id,
                "run.id":       state.run_id,
                "agent.id":     state.agent_id,
                "task.id":      state.task_id,
                "agent.name":   AGENT_NAME,
                "agent.stack":  AGENT_STACK,
            })
            if state.conversation_id:
                set_conversation_id(state.conversation_id)

            result = await client.call_tool(tool_name, arguments)
            if result is None:
                return f"ERROR: {tool_name} returned no response"
            return "\n".join(
                getattr(b, "text", "") for b in result.content if getattr(b, "text", None)
            )
    except ToolError as e:
        _log("ERROR", f"{tool_name}: {e}")
        return f"ERROR: {e}"
    except Exception as e:
        _log("ERROR", f"{tool_name}: {e}")
        return f"ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool factory
# ─────────────────────────────────────────────────────────────────────────────

def make_tools(state: RunState) -> list:

    async def register_agent(name: str, stack: str) -> str:
        """Register this agent in the Agent Arena. Call once at the start."""
        result = await _mcp_call("register_agent", {
            "idToken":     ID_TOKEN,
            "name":        name,
            "stack":       stack,
            "linkedinUrl": LINKEDIN_URL,
            "githubUrl":   GITHUB_URL,
        }, state)

        match = re.search(r"AGENT_ID:\s*(\S+?)\.?(\s|$)", result)
        if match:
            state.agent_id = match.group(1)
            state.conversation_id = state.agent_id
            set_association_properties({"agent.id": state.agent_id, "run.id": state.run_id})
            set_conversation_id(state.agent_id)

        level_match = re.search(r"Level[:\s]+(\d+)", result)
        if level_match:
            state.current_level = int(level_match.group(1))

        agent_logger.info("Registered", extra={"agent_id": state.agent_id, "run_id": state.run_id})
        _log("REGISTER", f"agent_id={state.agent_id}  level={state.current_level}")
        return result

    async def get_tasks(agent_id: str) -> str:
        """Fetch the currently assigned task for this agent's level."""
        result = await _mcp_call("get_tasks", {
            "idToken": ID_TOKEN, "agentId": agent_id,
        }, state)

        try:
            data = json.loads(result)
            # Handle both dict and list responses
            task_obj = None
            if isinstance(data, dict) and "id" in data:
                task_obj = data
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "id" in data[0]:
                task_obj = data[0]

            if task_obj:
                state.task_id         = task_obj["id"]
                state.current_task    = task_obj
                state.conversation_id = f"{state.agent_id}-{state.task_id}"
                set_association_properties({"task.id": state.task_id, "execution.id": state.execution_id})
                set_conversation_id(state.conversation_id)
                _log("FETCH", f"task={state.task_id}  '{task_obj.get('title')}'  L{task_obj.get('level')}")
        except json.JSONDecodeError:
            pass

        return result

    async def skip_task(agent_id: str, task_id: str, reason: str = "") -> str:
        """Abandon the current task and allow get_tasks to return a new one."""
        _log("SKIP", f"skipping {task_id[:8]}  reason={reason[:50]}")
        return await _mcp_call("skip_task", {
            "idToken": ID_TOKEN, "agentId": agent_id,
            "taskId": task_id, "reason": reason,
        }, state)

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """Submit the complete answer for the current task for AI evaluation."""
        new_exec = str(uuid.uuid4())
        state.execution_id = new_exec
        set_association_properties({
            "execution.id": new_exec,
            "task.id":      task_id,
            "agent.id":     agent_id,
        })

        task_logger.info("Submitting", extra={
            "agent_id": agent_id, "task_id": task_id, "execution_id": new_exec,
        })

        result = await _mcp_call("submit_task", {
            "idToken":     ID_TOKEN,
            "agentId":     agent_id,
            "taskId":      task_id,
            "executionId": new_exec,
            "content":     content,
            "metadata": {
                "agent_name": AGENT_NAME, "agent_stack": AGENT_STACK,
                "run_id": state.run_id, "execution_id": new_exec, "model": _active_model_name(),
            },
        }, state)

        score_match = re.search(r"Score:\s*(\d+)/100", result)
        score       = int(score_match.group(1)) if score_match else -1
        levelled_up = "LEVEL_UP" in result

        task_title = state.current_task.get("title", state.task_id) if state.current_task else state.task_id
        state.record(state.current_level, task_title, score, levelled_up)

        lu_emoji = "🚀 LEVEL_UP!" if levelled_up else ""
        _log("SCORE", f"{score}/100  {lu_emoji}")
        print(state.scoreboard())

        task_logger.info("Submitted", extra={
            "agent_id": agent_id, "task_id": task_id,
            "score": score, "levelled_up": levelled_up,
        })
        return result

    async def report_status() -> str:
        """Report the current agent status."""
        return (
            f"Agent: {AGENT_NAME}  ID: {state.agent_id}\n"
            f"Level: {state.current_level}  Total Score: {state.total_score}\n"
            f"Tasks attempted: {state.tasks_attempted}  Passed: {state.tasks_passed}\n"
            f"History: {json.dumps(state.level_history, indent=2)}"
        )

    return [register_agent, get_tasks, skip_task, submit_task, report_status]


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an expert autonomous agent competing in the Agent Arena evaluation system.
Your goal is to solve tasks with exceptional quality and advance through levels.

AVAILABLE TOOLS:
- register_agent(name, stack): Register once at the start.
- get_tasks(agent_id): Fetch the current task.
- skip_task(agent_id, task_id, reason): Skip an impossible/already-submitted task.
- submit_task(agent_id, task_id, content): Submit your final answer for evaluation.
- report_status(): Report progress before stopping.

CORE PRINCIPLES:
1. THOROUGHNESS: Analyze deeply, consider edge cases, and verify correctness.
2. QUALITY: Aim for 90+/100. Incomplete or shallow answers score poorly.
3. AUTONOMY: Do not ask for clarification. Make reasonable assumptions and state them.
4. ADAPTABILITY: Follow the specific instructions in each user message precisely.

RULES:
- Never submit the same task_id twice.
- Always use the task_id from the most recent get_tasks call.
- Do not ask for confirmation — act autonomously.
- When instructed to analyze+solve+submit, do all three in this turn.

IDENTITY:
- Agent Name: {AGENT_NAME}
- Stack: {AGENT_STACK}
""".strip()


def build_agent(state: RunState) -> LlmAgent:
    return LlmAgent(
        name="arena_agent",
        model=_active_model(),
        instruction=SYSTEM_PROMPT,
        tools=make_tools(state),
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_turn(
    runner:          Runner,
    session_service: InMemorySessionService,
    session_id:      str,
    message:         str,
) -> str:
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id="arena-user",
        session_id=session_id,
        new_message=content,
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args_str = str(dict(fc.args))
                preview  = args_str[:120]
                _log("AGENT", f"→ {fc.name}  {preview}{'...' if len(args_str) > 120 else ''}")

            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                resp_str = str(fr.response)[:150].replace("\n", " ")
                _log("AGENT", f"← {fr.name}  {resp_str}{'...' if len(str(fr.response)) > 150 else ''}")

            elif hasattr(part, "text") and part.text and event.turn_complete:
                final_text = part.text

    return final_text


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

@workflow(name="arena_adk_run")
async def run() -> None:
    state = RunState()

    print(f"\n{'═'*60}")
    print(f"  AGENT ARENA  —  {_active_model_name()}")
    print(f"{'═'*60}")
    _log("REGISTER", f"Agent: {AGENT_NAME}")
    _log("REGISTER", f"Run ID: {state.run_id}")
    _log("REGISTER", f"Max tasks: {MAX_TASKS}")
    print(f"{'═'*60}\n")

    set_association_properties({
        "run.id":       state.run_id,
        "execution.id": state.execution_id,
        "agent.name":   AGENT_NAME,
        "agent.stack":  AGENT_STACK,
    })

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="arena-adk-agent",
        user_id="arena-user",
        session_id=state.run_id,
    )

    agent  = build_agent(state)
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name="arena-adk-agent",
    )

    # ── Bootstrap: register and fetch first task ──────────────────────────────
    _log("REGISTER", "Bootstrapping — register then fetch first task...")
    await run_turn(
        runner, session_service, state.run_id,
        f"Call register_agent(name='{AGENT_NAME}', stack='{AGENT_STACK}') to register. "
        f"Then call get_tasks with your agent_id to fetch the first task. "
        f"Return ONLY a one-line summary: 'Task: <title> (Level <level>)'. "
        f"Do NOT solve or submit yet.",
    )

    if not state.current_task:
        _log("WARN", "No task after bootstrap. Attempting one more fetch...")
        await run_turn(
            runner, session_service, state.run_id,
            "Call get_tasks to fetch the first challenge.",
        )

    # ── Main task loop ────────────────────────────────────────────────────────
    for task_num in range(1, MAX_TASKS + 1):
        if not state.current_task or not state.task_id:
            _log("DONE", "No active task — stopping.")
            break

        task = state.current_task
        task_title = task.get("title", "Unknown")
        task_type  = detect_task_type(task_title, task.get("description", ""))
        desc       = task.get("description", "")[:600]

        print(f"\n{'━'*60}")
        _log("TASK", f"#{task_num} | {task_title}")
        _log("TASK", f"Type: {task_type.upper()} | Level: {task.get('level', '?')} | ID: {state.task_id[:8]}")
        _log("TASK", f"Desc: {desc}{'...' if len(task.get('description', '')) > 600 else ''}")
        print(f"{'━'*60}")

        # ── Single-turn solve ─────────────────────────────────────────────────
        prompt = build_task_prompt(task, state.agent_id, state.task_id)
        _log("AGENT", "Solving task (analysis + solution + submit in one turn)...")
        await run_turn(runner, session_service, state.run_id, prompt)

        # ── Verify submission ─────────────────────────────────────────────────
        prev_attempted = state.tasks_attempted
        if state.tasks_attempted > prev_attempted:
            _log("SCORE", f"Task #{task_num} submitted successfully.")
        else:
            _log("WARN", f"Task #{task_num} was NOT submitted. Recovering...")
            recovery = await run_turn(
                runner, session_service, state.run_id,
                f"You have NOT submitted the current task yet. "
                f"Call submit_task(agent_id='{state.agent_id}', task_id='{state.task_id}', "
                f"content=<your complete final answer>) NOW. "
                f"If the task is impossible to solve, call skip_task with a reason, then get_tasks.",
            )
            if state.tasks_attempted == prev_attempted:
                _log("ERROR", f"Recovery failed for task #{task_num}. Moving on.")
                # Force skip so we don't get stuck on the same task
                await run_turn(
                    runner, session_service, state.run_id,
                    f"Call skip_task(agent_id='{state.agent_id}', task_id='{state.task_id}', "
                    f"reason='Agent failed to submit after recovery prompt.')",
                )

        # ── Prepare for next task ─────────────────────────────────────────────
        state.current_task = None
        state.task_id = ""

        _log("LOOP", "Fetching next task...")
        await run_turn(
            runner, session_service, state.run_id,
            "Call get_tasks to fetch the next challenge. "
            "If NO_TASKS is returned, call report_status() and stop. "
            "Otherwise, return a brief summary.",
        )

        if not state.current_task:
            _log("DONE", "No more tasks available.")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    _log("DONE", "Final status report")
    print(f"{'═'*60}")
    await run_turn(
        runner, session_service, state.run_id,
        "Call report_status() to summarize your full run.",
    )
    print(state.scoreboard())
    agent_logger.info("Run complete", extra={
        "run_id":          state.run_id,
        "total_score":     state.total_score,
        "tasks_attempted": state.tasks_attempted,
        "final_level":     state.current_level,
    })


if __name__ == "__main__":
    init_tracing()
    asyncio.run(run())