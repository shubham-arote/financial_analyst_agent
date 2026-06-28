"""Calculator tests — correctness AND security.

The security half matters most: safe_eval parses untrusted LLM output, so it must reject
every form of code execution, name/attribute access, and resource abuse.
"""

import math

import pytest

from app.agent.calculator import (safe_eval, extract_expression, is_math_query, CalcError)


# ── correctness ───────────────────────────────────────────────────────────────
def test_basic_arithmetic():
    assert safe_eval("1052 - 985") == 67
    assert safe_eval("2 ** 8") == 256
    assert safe_eval("(10 + 5) * 2") == 30
    assert safe_eval("abs(-3)") == 3
    assert safe_eval("round(5.9 / 48.2 * 100, 1)") == 12.2
    assert safe_eval("max(1, 2, 3)") == 3


def test_yoy_growth_is_exact():
    # FY26 op profit 1052 vs FY25 985 -> -6.37% (the headline analyst example, sign as given)
    assert math.isclose(safe_eval("(985 - 1052) / 1052 * 100"), -6.3688, rel_tol=1e-3)


def test_commas_in_figures_are_stripped():
    assert safe_eval("1,052 - 985") == 67


# ── security: every one of these MUST raise, never execute ────────────────────
@pytest.mark.parametrize("evil", [
    "__import__('os').system('echo hi')",
    "os.system('rm -rf /')",
    "open('/etc/passwd')",
    "(1).__class__",
    "().__class__.__bases__",
    "x",                       # bare name
    "x + 1",
    "[i for i in range(9)]",   # comprehension
    "{1: 2}",                  # dict
    "lambda: 1",
    "1 if True else 2",        # IfExp
    "print(1)",
    "9 ** 9 ** 9",             # resource bomb -> capped
    "10 ** 9999",
    "eval('1+1')",
    "1; 2",                    # multiple statements (syntax error in eval mode)
    "import os",
])
def test_rejects_unsafe(evil):
    with pytest.raises(CalcError):
        safe_eval(evil)


def test_rejects_empty_and_oversized():
    with pytest.raises(CalcError):
        safe_eval("")
    with pytest.raises(CalcError):
        safe_eval("1+" * 200)


# ── expression extraction + intent ────────────────────────────────────────────
def test_extract_expression():
    assert extract_expression("NONE") is None
    assert extract_expression("(985-1052)/1052*100") == "(985-1052)/1052*100"
    # a clean numeric reply still evaluates
    assert safe_eval(extract_expression("1052 - 985")) == 67


def test_is_math_query():
    assert is_math_query("what was the YoY growth in operating profit?")
    assert is_math_query("what is the operating margin percentage?")
    assert not is_math_query("who is the chief executive?")
    assert not is_math_query("what is the registered office address?")
