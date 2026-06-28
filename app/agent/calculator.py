"""
calculator.py — deterministic arithmetic for the analyst agent.

The agent computes financial figures (growth %, margins, ratios) by having the LLM emit a
single arithmetic *expression* over retrieved numbers, which we then evaluate **exactly** here.
That keeps the arithmetic correct even when the language model is bad at mental math.

SECURITY: `safe_eval` parses an UNTRUSTED string (LLM output). It NEVER uses `eval`/`exec`.
It walks the AST and permits only numeric literals, the arithmetic operators, parentheses, and
a tiny whitelist of functions (abs/round/min/max). Names, attribute access, calls to anything
else, subscripts, comprehensions, etc. are rejected. `**` magnitude is capped to prevent a
`9**9**9`-style resource bomb.
"""

from __future__ import annotations

import ast
import operator
import re

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max}

_MAX_LEN = 200      # reject absurdly long expressions outright
_MAX_POW_EXP = 100  # cap exponent magnitude (anti-DoS)
_MAX_POW_BASE = 1e6


class CalcError(ValueError):
    """Raised when an expression is unsafe, malformed, or not pure arithmetic."""


def _ev(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _ev(node.body)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int — exclude it; only real numbers allowed
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"non-numeric literal: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _BINOPS:
            raise CalcError(f"operator not allowed: {op.__name__}")
        left, right = _ev(node.left), _ev(node.right)
        if op is ast.Pow and (abs(right) > _MAX_POW_EXP or abs(left) > _MAX_POW_BASE):
            raise CalcError("exponent magnitude too large")
        return _BINOPS[op](left, right)

    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _UNARY:
            raise CalcError(f"unary operator not allowed: {op.__name__}")
        return _UNARY[op](_ev(node.operand))

    if isinstance(node, ast.Call):
        # only a bare whitelisted name, positional numeric args, no kwargs/starargs
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalcError("function not allowed")
        if node.keywords:
            raise CalcError("keyword args not allowed")
        return _FUNCS[node.func.id](*[_ev(a) for a in node.args])

    raise CalcError(f"disallowed expression element: {type(node).__name__}")


def safe_eval(expr: str) -> float:
    """Evaluate a pure-arithmetic expression safely. Raises CalcError on anything unsafe.

    Commas in figures (e.g. ``1,052``) are stripped first so the LLM can quote document
    numbers verbatim.
    """
    # strip thousands separators only (comma before a 3-digit group), not arg commas
    expr = re.sub(r",(?=\d{3}(?:\D|$))", "", (expr or "").strip())
    if not expr:
        raise CalcError("empty expression")
    if len(expr) > _MAX_LEN:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"syntax error: {e}") from None
    return float(_ev(tree))


# A leading arithmetic span: digits/operators/parens/funcs only. Used to pull a clean
# expression out of an LLM reply that may include stray prose.
_EXPR_SPAN = re.compile(r"[-+0-9.,()*/%\s]+|abs|round|min|max")


def extract_expression(text: str) -> str | None:
    """Pull a candidate arithmetic expression from an LLM reply; None if there isn't one.

    Returns the longest contiguous arithmetic-looking span that actually evaluates.
    """
    text = (text or "").strip()
    if not text or text.upper().startswith("NONE"):
        return None
    # try the whole thing first, then progressively trimmed leading spans
    candidates = [text] + ["".join(re.findall(r"[-+0-9.,()*/%]|abs|round|min|max|\s", text)).strip()]
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            safe_eval(cand)
            return cand
        except CalcError:
            continue
    return None


_MATH_INTENT = re.compile(
    r"\b(growth|grow|increase|decrease|decline|change|difference|delta|ratio|margin|"
    r"percent|percentage|proportion|share|yoy|year[- ]over[- ]year|cagr|average|mean|"
    r"sum|total|times|multiple|how much (more|less|higher|lower)|compared? to|versus|\bvs\b)\b",
    re.I,
)


def is_math_query(question: str) -> bool:
    """Heuristic: does answering this need arithmetic (a computed figure), not just lookup?"""
    q = question or ""
    return bool(_MATH_INTENT.search(q)) or bool(re.search(r"%|\bper cent\b", q, re.I))
