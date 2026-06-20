"""
Dynamic prompt engine for the Agent Arena.
Provides task-type detection and a single strong composite prompt
that guides the agent to analyze, solve, and submit in one turn.
"""

from typing import Optional

# ── Task type detection ──────────────────────────────────────────────────────
TASK_PATTERNS = {
    "code": [
        "code", "function", "implement", "write a", "program", "script",
        "class", "algorithm", "api", "method", "library", "module", "package",
        "build", "create a", "develop", "application", "service", "endpoint",
    ],
    "debug": [
        "debug", "fix", "error", "bug", "issue", "broken", "fails",
        "exception", "traceback", "crash", "wrong", "incorrect", "not working",
        "repair", "resolve", "troubleshoot",
    ],
    "explain": [
        "explain", "describe", "what is", "how does", "why", "difference between",
        "concept", "theory", "overview", "introduction", "compare", "contrast",
        "elaborate", "clarify", "discuss",
    ],
    "optimize": [
        "optimize", "performance", "efficient", "slow", "bottleneck", "memory",
        "speed", "complexity", "scale", "improve", "faster", "latency",
        "throughput", "resource", "cache", "compress", "reduce",
    ],
    "design": [
        "design", "architecture", "system", "database schema", "pattern",
        "structure", "model", "diagram", "plan", "blueprint", "component",
        "microservice", "flow", "sequence", "entity relationship",
    ],
    "test": [
        "test", "unit test", "pytest", "assert", "coverage", "mock", "testing",
        "tdd", "spec", "validate", "verify", "bdd", "integration test",
        "regression", "benchmark",
    ],
    "data": [
        "data", "csv", "json", "sql", "query", "database", "etl", "pipeline",
        "transform", "clean", "analyze", "visualization", "chart", "pandas",
        "dataframe", "dataset",
    ],
    "security": [
        "security", "auth", "authentication", "authorization", "jwt", "oauth",
        "encrypt", "hash", "vulnerability", "sanitize", "xss", "csrf", "sql injection",
        "penetration", "secure",
    ],
}


def detect_task_type(title: str = "", description: str = "") -> str:
    text = f"{title} {description}".lower()
    scores = {}
    for task_type, keywords in TASK_PATTERNS.items():
        scores[task_type] = sum(1 for kw in keywords if kw in text)
    if not scores or max(scores.values(), default=0) == 0:
        return "general"
    return max(scores, key=scores.get)


# ── Prompt templates ─────────────────────────────────────────────────────────

def _format_task(task: dict) -> str:
    lines = [
        f"Title: {task.get('title', 'N/A')}",
        f"Level: {task.get('level', 'N/A')}",
        f"Points: {task.get('points', 'N/A')}",
        f"Difficulty: {task.get('difficulty', 'N/A')}",
        f"Description:\n{task.get('description', 'N/A')}",
    ]
    return "\n".join(lines)


def build_task_prompt(task: dict, agent_id: str, task_id: str) -> str:
    """
    Single composite prompt that instructs the agent to:
      1. Analyze the task deeply
      2. Produce a complete, high-quality solution
      3. Call submit_task with the full solution
    """
    task_type = detect_task_type(task.get("title", ""), task.get("description", ""))

    type_guidance = {
        "code": (
            "Write clean, well-commented code with docstrings, type hints, error handling, "
            "and a brief usage example. Explain key design decisions."
        ),
        "debug": (
            "Identify the root cause, provide the fixed code/config, explain why the fix works, "
            "and suggest prevention strategies."
        ),
        "explain": (
            "Use clear analogies, step-by-step breakdowns, concrete examples, and address "
            "common misconceptions. Structure from simple to complex."
        ),
        "optimize": (
            "Show before/after reasoning, provide the optimized solution, explain performance gains, "
            "and note any trade-offs."
        ),
        "design": (
            "Provide architecture overview, component breakdown, data flow, technology choices "
            "with justification, and scalability/failure considerations."
        ),
        "test": (
            "Provide a complete test suite with setup, positive/negative cases, edge cases, "
            "and mocking strategy where applicable."
        ),
        "data": (
            "Provide the data pipeline/transformation logic, schema, validation checks, "
            "and sample outputs."
        ),
        "security": (
            "Summarize threat model, provide secure code/config, explain defense mechanisms, "
            "and include verification steps."
        ),
        "general": (
            "Provide a complete, well-structured, and thorough solution with examples and reasoning."
        ),
    }

    guidance = type_guidance.get(task_type, type_guidance["general"])

    return f"""
You have been assigned a new task. Solve it completely in this turn.

TASK ({task_type.upper()}):
{_format_task(task)}

REASONING & SOLVING INSTRUCTIONS:
1. ANALYZE — Restate the problem, extract all explicit and implicit requirements, list edge cases and risks, and outline your chosen approach with justification.
2. SOLVE — Produce a complete, production-ready answer. {guidance}
3. REVIEW — Before submitting, mentally verify: correctness, completeness, edge cases, clarity, best practices, and efficiency.

SUBMISSION INSTRUCTIONS:
After your analysis and solution, you MUST call submit_task with:
  agent_id = "{agent_id}"
  task_id  = "{task_id}"
  content  = <your complete final answer (analysis + solution)>

The content should be thorough and self-contained — the evaluator scores on correctness, depth, and robustness. Aim for 90+/100.

Do NOT stop after analysis. Do NOT ask for confirmation. Call submit_task in this same turn.
""".strip()