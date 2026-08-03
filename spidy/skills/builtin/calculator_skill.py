"""
CalculatorSkill — Safe Math Evaluation
=======================================
Provides arithmetic and expression evaluation WITHOUT using eval().

Permission Tier: T0 (read-only, no system changes)

Actions
-------
calculate    — Evaluate a math expression safely           (T0)

Safety
------
Uses Python's ``ast`` module to parse expressions, then evaluates
only a whitelisted set of node types. This prevents code injection,
file access, and any side effects.

Supported
---------
- Basic arithmetic: +, -, *, /, //, %, **
- Parentheses, order of operations
- Unary minus/plus
- Integer and float literals
- Natural language: "plus", "minus", "times", "divided by", "percent of"
- Percentage: "15 percent of 200" → 30.0
- Square root: "square root of 16" → 4.0

Examples
--------
    "calculate 15*32"          → "15 × 32 = 480"
    "what is 100 / 4"          → "100 ÷ 4 = 25.0"
    "15 percent of 200"        → "15% of 200 = 30.0"
    "square root of 144"       → "√144 = 12.0"
    "2 ** 10"                  → "2 ^ 10 = 1024"
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from spidy.skills.base import BaseSkill, ParamSchema, SkillCapability, SkillContext, SkillResult


# ─── Safe evaluator ───────────────────────────────────────────────────────────

# Whitelist of allowed AST node types
_SAFE_NODES = frozenset({
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
})

# Operator dispatch
_BINOP_MAP: dict[type, Any] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}

_UNOP_MAP: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted AST expression."""
    if type(node) not in _SAFE_NODES:
        raise ValueError(f"Unsupported operation: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_func = _BINOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        if isinstance(node.op, ast.FloorDiv) and right == 0:
            raise ZeroDivisionError("Division by zero")
        # Cap exponentiation to prevent DoS
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return float(op_func(left, right))

    if isinstance(node, ast.UnaryOp):
        op_func = _UNOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return float(op_func(_safe_eval(node.operand)))

    raise ValueError(f"Unexpected node type: {type(node).__name__}")


# ─── Natural language → expression normalization ──────────────────────────────

# Map spoken words to math symbols
_NL_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btimes\b",               re.I), "*"),
    (re.compile(r"\bmultiplied\s+by\b",     re.I), "*"),
    (re.compile(r"\bdivided\s+by\b",        re.I), "/"),
    (re.compile(r"\bover\b",                re.I), "/"),
    (re.compile(r"\bplus\b",                re.I), "+"),
    (re.compile(r"\bminus\b",               re.I), "-"),
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.I), "**"),
    (re.compile(r"\bpower\s+of\b",          re.I), "**"),
    (re.compile(r"\bmod\b",                 re.I), "%"),
    (re.compile(r"\bmodulo\b",              re.I), "%"),
    (re.compile(r"\bmodulus\b",             re.I), "%"),
    (re.compile(r"\bfloor\s+divide\b",      re.I), "//"),
]

_PERCENT_OF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*percent\s+of\s+(\d+(?:\.\d+)?)", re.I)
_SQRT_RE = re.compile(r"(?:square\s+root|sqrt)\s+(?:of\s+)?(\d+(?:\.\d+)?)", re.I)


def _normalize_expression(raw: str) -> tuple[str, str]:
    """
    Normalize a natural-language math expression to a Python expression string.

    Returns
    -------
    (normalized_expr, display_expr)
        normalized_expr: Python-parseable string (e.g. "15 * 32")
        display_expr: Human-readable version (e.g. "15 × 32")
    """
    raw = raw.strip()

    # Handle "X percent of Y"
    m = _PERCENT_OF_RE.search(raw)
    if m:
        pct, total = m.group(1), m.group(2)
        expr = f"({pct} / 100) * {total}"
        display = f"{pct}% of {total}"
        return expr, display

    # Handle "square root of X"
    m = _SQRT_RE.search(raw)
    if m:
        num = m.group(1)
        expr = f"{num} ** 0.5"
        display = f"√{num}"
        return expr, display

    # Apply word replacements
    normalized = raw
    for pattern, replacement in _NL_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)

    display = normalized
    return normalized, display


def _format_result(value: float, display_expr: str) -> str:
    """Format the calculation result for display."""
    # Show as integer if it's a whole number
    if value == int(value) and abs(value) < 1e15:
        result_str = str(int(value))
    else:
        result_str = f"{value:.6g}"  # up to 6 significant figures

    return f"{display_expr} = {result_str}"


# ─── Skill ────────────────────────────────────────────────────────────────────


class CalculatorSkill(BaseSkill):
    """
    Safe math evaluation skill.

    Uses a whitelisted AST evaluator — no eval(), no exec(),
    no file access, no network access.
    """

    name = "calculator_skill"
    version = "1.0.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="calculate",
                description=(
                    "Evaluate a mathematical expression or calculation safely. "
                    "Supports arithmetic, percentages, square roots, and natural language."
                ),
                permission_tier="T0",
                params=[
                    ParamSchema(
                        "expression",
                        "string",
                        required=True,
                        description="The math expression to evaluate (e.g. '15 * 32', '15 percent of 200').",
                    ),
                ],
                examples=[
                    "calculate 15*32",
                    "what is 100 divided by 4",
                    "15 percent of 200",
                    "square root of 144",
                    "2 to the power of 10",
                    "what is 5 plus 3 minus 2",
                    "compute 999 * 999",
                ],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "calculate":
            return await self._calculate(context)
        return SkillResult.fail(f"CalculatorSkill: unknown action '{action}'.")

    async def _calculate(self, context: SkillContext) -> SkillResult:
        raw_expr: str = context.get("expression", "") or ""
        if not raw_expr.strip():
            return SkillResult.fail(
                "Please provide a mathematical expression to evaluate. "
                "Example: 'calculate 15 * 32' or 'what is 100 divided by 4'."
            )

        try:
            normalized, display = _normalize_expression(raw_expr)
        except Exception as exc:
            return SkillResult.fail(
                f"Could not understand the expression '{raw_expr}': {exc}"
            )

        # Try to parse and evaluate
        try:
            tree = ast.parse(normalized, mode="eval")
            value = _safe_eval(tree)
        except ZeroDivisionError:
            return SkillResult.fail(
                "Cannot divide by zero.",
                error=ZeroDivisionError("Division by zero"),
            )
        except (SyntaxError, ValueError) as exc:
            return SkillResult.fail(
                f"I couldn't evaluate '{raw_expr}'. "
                f"Please check the expression and try again. ({exc})"
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"Calculation error: {exc}",
                error=exc,
            )

        result_msg = _format_result(value, display)
        return SkillResult.ok(
            result_msg,
            data={"expression": raw_expr, "normalized": normalized, "result": value},
            action_taken="calculate",
        )
