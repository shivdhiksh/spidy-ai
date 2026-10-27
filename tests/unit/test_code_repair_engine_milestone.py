"""
Tests for Code Agent and Bounded Error Repair Engine (Phase F)
"""

import pytest
import os
import sys
import tempfile
from spidy.agent.code_agent import CodeRepairEngine, DiagnosticInfo


def test_parse_error_output_python():
    tb = """
Traceback (most recent call last):
  File "C:\\projects\\app.py", line 42, in calculate_tax
    return income * rate
NameError: name 'rate' is not defined
"""
    diag = CodeRepairEngine.parse_error_output(tb)
    assert diag.error_type == "NameError"
    assert "rate" in diag.error_message
    assert diag.line_number == 42
    assert "app.py" in diag.file_path


def test_parse_error_output_pytest():
    tb = """
FAILED tests/test_math.py::test_add - AssertionError: assert 4 == 5
"""
    diag = CodeRepairEngine.parse_error_output(tb)
    assert diag.error_type == "AssertionError"
    assert "test_math.py" in diag.file_path


@pytest.mark.asyncio
async def test_code_repair_engine_deterministic_fix():
    with tempfile.TemporaryDirectory() as tmpdir:
        broken_file = os.path.join(tmpdir, "broken.py")
        with open(broken_file, "w", encoding="utf-8") as f:
            f.write("print(undefined_variable)\n")

        engine = CodeRepairEngine(llm_client=None, max_repair_attempts=2)
        initial_error = f'File "{broken_file}", line 1, in <module>\nNameError: name \'undefined_variable\' is not defined'

        # Command runs python on the script
        cmd = f'"{sys.executable}" "{broken_file}"'
        success, msg, attempts = await engine.repair_and_verify(
            command_or_test=cmd,
            initial_error=initial_error,
            target_file=broken_file,
            cwd=tmpdir,
        )

        assert success is True
        assert len(attempts) >= 1
        with open(broken_file, "r", encoding="utf-8") as f:
            fixed_code = f.read()
        assert "undefined_variable = 0" in fixed_code
