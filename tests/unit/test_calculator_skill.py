"""
tests/unit/test_calculator_skill.py — CalculatorSkill Test Suite
=================================================================
Tests for safe AST-based math evaluation, natural language normalization,
edge cases, and error handling.
"""

from __future__ import annotations

import pytest

from spidy.skills.builtin.calculator_skill import (
    CalculatorSkill,
    _normalize_expression,
    _format_result,
    _safe_eval,
)
from spidy.skills.base import SkillContext


# ─── Unit Tests for Internal Functions ────────────────────────────────────────

class TestNormalizeExpression:
    def test_plain_arithmetic(self):
        expr, display = _normalize_expression("15 * 32")
        assert expr == "15 * 32"

    def test_times_word(self):
        expr, display = _normalize_expression("5 times 3")
        assert "*" in expr

    def test_divided_by(self):
        expr, display = _normalize_expression("100 divided by 4")
        assert "/" in expr

    def test_plus_word(self):
        expr, display = _normalize_expression("5 plus 3")
        assert "+" in expr

    def test_minus_word(self):
        expr, display = _normalize_expression("10 minus 3")
        assert "-" in expr

    def test_percent_of(self):
        expr, display = _normalize_expression("15 percent of 200")
        # Should produce a fraction expression
        assert "100" in expr
        assert display == "15% of 200"

    def test_sqrt(self):
        expr, display = _normalize_expression("square root of 144")
        assert "0.5" in expr
        assert "√144" in display

    def test_power_of(self):
        expr, display = _normalize_expression("2 to the power of 10")
        assert "**" in expr

    def test_multiplied_by(self):
        expr, display = _normalize_expression("7 multiplied by 8")
        assert "*" in expr


class TestSafeEval:
    """Test the AST evaluator directly."""

    def test_addition(self):
        import ast
        tree = ast.parse("2 + 3", mode="eval")
        assert _safe_eval(tree) == 5.0

    def test_subtraction(self):
        import ast
        tree = ast.parse("10 - 4", mode="eval")
        assert _safe_eval(tree) == 6.0

    def test_multiplication(self):
        import ast
        tree = ast.parse("15 * 32", mode="eval")
        assert _safe_eval(tree) == 480.0

    def test_division(self):
        import ast
        tree = ast.parse("100 / 4", mode="eval")
        assert _safe_eval(tree) == 25.0

    def test_power(self):
        import ast
        tree = ast.parse("2 ** 10", mode="eval")
        assert _safe_eval(tree) == 1024.0

    def test_order_of_operations(self):
        import ast
        tree = ast.parse("2 + 3 * 4", mode="eval")
        assert _safe_eval(tree) == 14.0

    def test_parentheses(self):
        import ast
        tree = ast.parse("(2 + 3) * 4", mode="eval")
        assert _safe_eval(tree) == 20.0

    def test_unary_negative(self):
        import ast
        tree = ast.parse("-5 + 10", mode="eval")
        assert _safe_eval(tree) == 5.0

    def test_floor_division(self):
        import ast
        tree = ast.parse("10 // 3", mode="eval")
        assert _safe_eval(tree) == 3.0

    def test_modulo(self):
        import ast
        tree = ast.parse("10 % 3", mode="eval")
        assert _safe_eval(tree) == 1.0

    def test_division_by_zero_raises(self):
        import ast
        tree = ast.parse("5 / 0", mode="eval")
        with pytest.raises(ZeroDivisionError):
            _safe_eval(tree)

    def test_string_literal_rejected(self):
        import ast
        tree = ast.parse("'hello'", mode="eval")
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(tree)

    def test_function_call_rejected(self):
        import ast
        tree = ast.parse("print(1)", mode="eval")
        with pytest.raises((ValueError, TypeError)):
            _safe_eval(tree)

    def test_large_exponent_rejected(self):
        import ast
        tree = ast.parse("2 ** 9999", mode="eval")
        with pytest.raises(ValueError, match="too large"):
            _safe_eval(tree)


class TestFormatResult:
    def test_whole_number(self):
        result = _format_result(480.0, "15 * 32")
        assert "480" in result
        assert "." not in result  # should not show 480.0

    def test_float(self):
        result = _format_result(25.5, "51 / 2")
        assert "25.5" in result

    def test_expression_in_output(self):
        result = _format_result(10.0, "5 + 5")
        assert "5 + 5" in result
        assert "10" in result


# ─── Skill Integration Tests ───────────────────────────────────────────────────

def make_ctx(params: dict) -> SkillContext:
    return SkillContext(action="calculate", params=params, session_id="test", user_name="Test")


class TestCalculatorSkill:
    @pytest.fixture
    def skill(self):
        return CalculatorSkill()

    @pytest.mark.asyncio
    async def test_basic_multiply(self, skill):
        ctx = make_ctx({"expression": "15 * 32"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "480" in result.message

    @pytest.mark.asyncio
    async def test_basic_add(self, skill):
        ctx = make_ctx({"expression": "5 + 3"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "8" in result.message

    @pytest.mark.asyncio
    async def test_division(self, skill):
        ctx = make_ctx({"expression": "100 / 4"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "25" in result.message

    @pytest.mark.asyncio
    async def test_percent_of(self, skill):
        ctx = make_ctx({"expression": "15 percent of 200"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "30" in result.message

    @pytest.mark.asyncio
    async def test_square_root(self, skill):
        ctx = make_ctx({"expression": "square root of 144"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "12" in result.message

    @pytest.mark.asyncio
    async def test_power(self, skill):
        ctx = make_ctx({"expression": "2 to the power of 10"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "1024" in result.message

    @pytest.mark.asyncio
    async def test_natural_language_times(self, skill):
        ctx = make_ctx({"expression": "7 times 8"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "56" in result.message

    @pytest.mark.asyncio
    async def test_division_by_zero(self, skill):
        ctx = make_ctx({"expression": "5 / 0"})
        result = await skill.execute("calculate", ctx)
        assert not result.success
        assert "zero" in result.message.lower()

    @pytest.mark.asyncio
    async def test_empty_expression(self, skill):
        ctx = make_ctx({"expression": ""})
        result = await skill.execute("calculate", ctx)
        assert not result.success
        assert "expression" in result.message.lower()

    @pytest.mark.asyncio
    async def test_no_expression_param(self, skill):
        ctx = make_ctx({})
        result = await skill.execute("calculate", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_data_contains_result(self, skill):
        ctx = make_ctx({"expression": "3 + 4"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert result.data is not None
        assert result.data["result"] == 7.0

    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        ctx = make_ctx({})
        result = await skill.execute("do_magic", ctx)
        assert not result.success
        assert "unknown action" in result.message.lower()

    @pytest.mark.asyncio
    async def test_order_of_operations(self, skill):
        ctx = make_ctx({"expression": "2 + 3 * 4"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        assert "14" in result.message

    @pytest.mark.asyncio
    async def test_complex_expression(self, skill):
        ctx = make_ctx({"expression": "(100 + 50) * 2 / 3"})
        result = await skill.execute("calculate", ctx)
        assert result.success
        # 150 * 2 / 3 = 100
        assert "100" in result.message

    @pytest.mark.asyncio
    async def test_capabilities_registered(self, skill):
        caps = skill.capabilities()
        assert len(caps) == 1
        assert caps[0].action == "calculate"

    @pytest.mark.asyncio
    async def test_supports_calculate_action(self, skill):
        assert skill.supports("calculate")

    @pytest.mark.asyncio
    async def test_does_not_support_unknown(self, skill):
        assert not skill.supports("launch_app")
