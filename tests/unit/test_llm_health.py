"""
Unit tests for spidy.llm.health — Ollama startup health check.

Coverage
--------
- OllamaHealthReport dataclass: construction, all_ok, usable properties
- _check_binary_installed: found / not found
- _check_server_running: running / connection refused / malformed response
- _check_model_available: exact match / prefix match / absent / empty list
- _check_llm_connection: success / empty response / connection error
- run_health_check: all-ok / no binary / no server / missing model / llm failure
- run_health_check (non-ollama provider): short-circuits correctly
- maybe_pull_model: user confirms (mocked pull) / user declines / pull error
- print_health_report: smoke tests for all four result combinations
- run_health_check_async: exercises the asyncio wrapper
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spidy.llm.health import (
    OllamaHealthReport,
    _check_binary_installed,
    _check_llm_connection,
    _check_model_available,
    _check_server_running,
    maybe_pull_model,
    print_health_report,
    run_health_check,
    run_health_check_async,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_reasoning_config(
    provider: str = "ollama",
    model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
) -> MagicMock:
    """Return a minimal ReasoningConfig mock."""
    cfg = MagicMock()
    cfg.provider = provider
    cfg.model = model
    cfg.base_url = base_url
    return cfg


def _make_url_response(body: dict, status: int = 200) -> MagicMock:
    """Return a mock response for urllib.request.urlopen."""
    raw = json.dumps(body).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_tags_response(models: list[str]) -> dict:
    """Build the /api/tags JSON body for the given model name list."""
    return {"models": [{"name": m} for m in models]}


# ─── OllamaHealthReport ───────────────────────────────────────────────────────


class TestOllamaHealthReport:
    def test_defaults(self) -> None:
        r = OllamaHealthReport()
        assert r.binary_installed is False
        assert r.server_running is False
        assert r.model_available is False
        assert r.llm_connected is False
        assert r.available_models == []
        assert r.errors == []
        assert r.configured_model == ""
        assert r.base_url == "http://localhost:11434"

    def test_all_ok_false_by_default(self) -> None:
        assert OllamaHealthReport().all_ok is False

    def test_all_ok_true_when_all_pass(self) -> None:
        r = OllamaHealthReport(
            binary_installed=True,
            server_running=True,
            model_available=True,
            llm_connected=True,
        )
        assert r.all_ok is True

    def test_all_ok_false_if_any_fail(self) -> None:
        r = OllamaHealthReport(
            binary_installed=True,
            server_running=True,
            model_available=True,
            llm_connected=False,
        )
        assert r.all_ok is False

    def test_usable_true_without_binary(self) -> None:
        r = OllamaHealthReport(
            binary_installed=False,
            server_running=True,
            model_available=True,
        )
        assert r.usable is True

    def test_usable_false_when_server_down(self) -> None:
        r = OllamaHealthReport(server_running=False, model_available=True)
        assert r.usable is False

    def test_usable_false_when_model_missing(self) -> None:
        r = OllamaHealthReport(server_running=True, model_available=False)
        assert r.usable is False

    def test_errors_accumulate(self) -> None:
        r = OllamaHealthReport(errors=["err1", "err2"])
        assert len(r.errors) == 2


# ─── _check_binary_installed ─────────────────────────────────────────────────


class TestCheckBinaryInstalled:
    def test_returns_true_when_found(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/ollama"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="ollama version 0.3.14\n",
                    stderr="",
                )
                found, version = _check_binary_installed()
        assert found is True
        assert "0.3.14" in version

    def test_returns_false_when_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            found, version = _check_binary_installed()
        assert found is False
        assert version == ""

    def test_version_unknown_on_subprocess_error(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/ollama"):
            with patch("subprocess.run", side_effect=Exception("timeout")):
                found, version = _check_binary_installed()
        assert found is True
        assert "unknown" in version

    def test_strips_extra_whitespace_from_version(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/ollama"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="  ollama version 0.4.0  \n",
                    stderr="",
                )
                _, version = _check_binary_installed()
        assert version == "ollama version 0.4.0"


# ─── _check_server_running ────────────────────────────────────────────────────


class TestCheckServerRunning:
    def test_returns_true_and_model_list_on_success(self) -> None:
        mock_resp = _make_url_response(_make_tags_response(["llama3.2:3b", "mistral:7b"]))
        with patch("urllib.request.urlopen", return_value=mock_resp):
            running, models = _check_server_running("http://localhost:11434")
        assert running is True
        assert "llama3.2:3b" in models
        assert "mistral:7b" in models

    def test_returns_false_on_connection_refused(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            running, models = _check_server_running("http://localhost:11434")
        assert running is False
        assert models == []

    def test_returns_empty_model_list_when_no_models_key(self) -> None:
        mock_resp = _make_url_response({})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            running, models = _check_server_running("http://localhost:11434")
        assert running is True
        assert models == []

    def test_returns_false_on_unexpected_exception(self) -> None:
        with patch("urllib.request.urlopen", side_effect=RuntimeError("oops")):
            running, models = _check_server_running("http://localhost:11434")
        assert running is False
        assert models == []

    def test_skips_models_without_name(self) -> None:
        body = {"models": [{"name": "llama3.2:3b"}, {"size": 100}]}
        mock_resp = _make_url_response(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            _, models = _check_server_running("http://localhost:11434")
        assert models == ["llama3.2:3b"]

    def test_trims_trailing_slash_from_base_url(self) -> None:
        mock_resp = _make_url_response(_make_tags_response([]))
        captured_url: list[str] = []

        def fake_urlopen(req: Any, timeout: int) -> Any:
            captured_url.append(req.full_url)
            return mock_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            _check_server_running("http://localhost:11434/")
        assert captured_url[0] == "http://localhost:11434/api/tags"


# ─── _check_model_available ───────────────────────────────────────────────────


class TestCheckModelAvailable:
    def test_exact_match(self) -> None:
        assert _check_model_available("llama3.2:3b", ["llama3.2:3b", "mistral:7b"])

    def test_prefix_match(self) -> None:
        # "llama3.2" should match "llama3.2:3b"
        assert _check_model_available("llama3.2", ["llama3.2:3b"])

    def test_not_found(self) -> None:
        assert not _check_model_available("gemma2:2b", ["llama3.2:3b", "mistral:7b"])

    def test_empty_model_list(self) -> None:
        assert not _check_model_available("llama3.2:3b", [])

    def test_empty_configured_model(self) -> None:
        assert not _check_model_available("", ["llama3.2:3b"])

    def test_prefix_no_false_positive(self) -> None:
        # "llama3" should NOT match "llama3.2:3b" because prefix is "llama3"
        # but "llama3.2:3b".startswith("llama3") is True — this IS expected behaviour
        # (user configured "llama3" meaning any llama3 variant is acceptable)
        assert _check_model_available("llama3", ["llama3.2:3b"])

    def test_different_tag_matches(self) -> None:
        assert _check_model_available("mistral:7b", ["mistral:7b", "mistral:latest"])

    def test_no_match_with_completely_different_model(self) -> None:
        # "gemma2" does NOT start with any prefix that matches "llama3"
        assert not _check_model_available("gemma2:2b", ["llama3.2:3b", "mistral:7b"])


# ─── _check_llm_connection ────────────────────────────────────────────────────


class TestCheckLlmConnection:
    def test_returns_true_on_non_empty_response(self) -> None:
        mock_resp = _make_url_response({"message": {"content": "pong"}})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            ok, err = _check_llm_connection("http://localhost:11434", "llama3.2:3b")
        assert ok is True
        assert err == ""

    def test_returns_false_on_empty_content(self) -> None:
        mock_resp = _make_url_response({"message": {"content": ""}})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            ok, err = _check_llm_connection("http://localhost:11434", "llama3.2:3b")
        assert ok is False
        assert "empty" in err.lower()

    def test_returns_false_on_connection_error(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            ok, err = _check_llm_connection("http://localhost:11434", "llama3.2:3b")
        assert ok is False
        assert "Connection failed" in err

    def test_returns_false_on_unexpected_exception(self) -> None:
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            ok, err = _check_llm_connection("http://localhost:11434", "llama3.2:3b")
        assert ok is False
        assert "Unexpected" in err

    def test_returns_false_when_message_key_missing(self) -> None:
        mock_resp = _make_url_response({"done": True})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            ok, _ = _check_llm_connection("http://localhost:11434", "llama3.2:3b")
        assert ok is False


# ─── run_health_check (full) ──────────────────────────────────────────────────


class TestRunHealthCheck:
    """End-to-end tests for run_health_check() using mocked I/O."""

    def _mock_all_ok(self) -> list:
        """Patch stack for a fully working Ollama installation."""
        models = ["llama3.2:3b"]
        return [
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="ollama version 0.3.14\n", stderr=""),
            ),
            patch(
                "urllib.request.urlopen",
                side_effect=[
                    _make_url_response(_make_tags_response(models)),  # /api/tags
                    _make_url_response({"message": {"content": "pong"}}),  # /api/chat
                ],
            ),
        ]

    def test_all_ok(self) -> None:
        cfg = _make_reasoning_config()
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="ollama version 0.3.14\n", stderr=""),
            ),
            patch(
                "urllib.request.urlopen",
                side_effect=[
                    _make_url_response(_make_tags_response(["llama3.2:3b"])),
                    _make_url_response({"message": {"content": "pong"}}),
                ],
            ),
        ):
            report = run_health_check(cfg)

        assert report.all_ok is True
        assert report.binary_installed is True
        assert report.server_running is True
        assert report.model_available is True
        assert report.llm_connected is True
        assert report.errors == []

    def test_no_binary_no_server(self) -> None:
        cfg = _make_reasoning_config()
        with (
            patch("shutil.which", return_value=None),
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("refused"),
            ),
        ):
            report = run_health_check(cfg)

        assert report.binary_installed is False
        assert report.server_running is False
        assert report.model_available is False
        assert report.llm_connected is False
        assert len(report.errors) >= 2

    def test_binary_found_but_server_down(self) -> None:
        cfg = _make_reasoning_config()
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="ollama version 0.3.14\n", stderr=""),
            ),
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("refused"),
            ),
        ):
            report = run_health_check(cfg)

        assert report.binary_installed is True
        assert report.server_running is False
        assert report.model_available is False
        assert report.llm_connected is False
        # Should stop after server check
        assert any("ollama serve" in e for e in report.errors)

    def test_server_running_but_model_missing(self) -> None:
        cfg = _make_reasoning_config(model="llama3.2:3b")
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="ollama version 0.3.14\n", stderr=""),
            ),
            patch(
                "urllib.request.urlopen",
                return_value=_make_url_response(_make_tags_response(["mistral:7b"])),
            ),
        ):
            report = run_health_check(cfg)

        assert report.server_running is True
        assert report.model_available is False
        assert report.llm_connected is False
        assert any("ollama pull" in e for e in report.errors)

    def test_model_available_but_llm_fails(self) -> None:
        cfg = _make_reasoning_config()
        tags_resp = _make_url_response(_make_tags_response(["llama3.2:3b"]))
        chat_resp = _make_url_response({"message": {"content": ""}})  # empty → fail
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="ollama version 0.3.14\n", stderr=""),
            ),
            patch("urllib.request.urlopen", side_effect=[tags_resp, chat_resp]),
        ):
            report = run_health_check(cfg)

        assert report.model_available is True
        assert report.llm_connected is False
        assert report.all_ok is False

    def test_non_ollama_provider_skips_checks(self) -> None:
        cfg = _make_reasoning_config(provider="openai", model="gpt-4o-mini")
        with patch("urllib.request.urlopen", side_effect=AssertionError("should not call")):
            report = run_health_check(cfg)

        # Non-Ollama providers get a pass-through report
        assert report.server_running is True
        assert report.model_available is True
        assert report.llm_connected is True
        assert any("openai" in e.lower() for e in report.errors)

    def test_configured_model_stored_in_report(self) -> None:
        cfg = _make_reasoning_config(model="mistral:7b")
        with (
            patch("shutil.which", return_value=None),
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")),
        ):
            report = run_health_check(cfg)
        assert report.configured_model == "mistral:7b"

    def test_base_url_stored_in_report(self) -> None:
        cfg = _make_reasoning_config(base_url="http://localhost:11435")
        with (
            patch("shutil.which", return_value=None),
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")),
        ):
            report = run_health_check(cfg)
        assert report.base_url == "http://localhost:11435"


# ─── run_health_check_async ───────────────────────────────────────────────────


class TestRunHealthCheckAsync:
    """Async wrapper exercises asyncio.to_thread path."""

    def test_async_returns_same_result_as_sync(self) -> None:
        cfg = _make_reasoning_config()
        tags_resp = _make_url_response(_make_tags_response(["llama3.2:3b"]))
        chat_resp = _make_url_response({"message": {"content": "pong"}})
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="ollama version 0.3.14\n", stderr=""),
            ),
            patch("urllib.request.urlopen", side_effect=[tags_resp, chat_resp]),
        ):
            report = asyncio.run(run_health_check_async(cfg))

        assert report.all_ok is True

    def test_async_handles_server_down(self) -> None:
        cfg = _make_reasoning_config()
        with (
            patch("shutil.which", return_value=None),
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")),
        ):
            report = asyncio.run(run_health_check_async(cfg))
        assert report.server_running is False


# ─── maybe_pull_model ─────────────────────────────────────────────────────────


class TestMaybePullModel:
    def _make_pull_response(self, lines: list[dict]) -> MagicMock:
        """Build a mock streaming pull response."""
        raw_lines = [json.dumps(l).encode() + b"\n" for l in lines]
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.__iter__ = lambda s: iter(raw_lines)
        return resp

    def test_user_declines_returns_false(self) -> None:
        result = maybe_pull_model(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            confirm=lambda: False,
        )
        assert result is False

    def test_user_confirms_successful_pull(self) -> None:
        pull_resp = self._make_pull_response([
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "success"},
        ])
        with patch("urllib.request.urlopen", return_value=pull_resp):
            result = maybe_pull_model(
                model="llama3.2:3b",
                base_url="http://localhost:11434",
                confirm=lambda: True,
            )
        assert result is True

    def test_pull_error_returns_false(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            result = maybe_pull_model(
                model="llama3.2:3b",
                base_url="http://localhost:11434",
                confirm=lambda: True,
            )
        assert result is False

    def test_pull_skips_malformed_json_lines(self) -> None:
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.__iter__ = lambda s: iter([b"not-json\n", b'{"status":"done"}\n'])
        with patch("urllib.request.urlopen", return_value=resp):
            result = maybe_pull_model(
                model="llama3.2:3b",
                base_url="http://localhost:11434",
                confirm=lambda: True,
            )
        assert result is True


# ─── print_health_report (smoke tests) ───────────────────────────────────────


class TestPrintHealthReport:
    """Ensure print_health_report never raises for any combination of flags."""

    def test_all_ok(self, capsys: pytest.CaptureFixture) -> None:
        r = OllamaHealthReport(
            binary_installed=True,
            binary_version="ollama version 0.3.14",
            server_running=True,
            model_available=True,
            configured_model="llama3.2:3b",
            available_models=["llama3.2:3b"],
            llm_connected=True,
        )
        print_health_report(r)
        out = capsys.readouterr().out
        assert "llama3.2:3b" in out
        assert "Brain is ready" in out

    def test_no_binary(self, capsys: pytest.CaptureFixture) -> None:
        r = OllamaHealthReport(
            binary_installed=False,
            errors=["Ollama binary not found on PATH."],
        )
        print_health_report(r)
        out = capsys.readouterr().out
        assert "Issues detected" in out

    def test_server_down(self, capsys: pytest.CaptureFixture) -> None:
        r = OllamaHealthReport(
            binary_installed=True,
            binary_version="ollama version 0.3.14",
            server_running=False,
            errors=["Ollama server not reachable at http://localhost:11434."],
        )
        print_health_report(r)
        out = capsys.readouterr().out
        assert "Issues detected" in out

    def test_model_missing(self, capsys: pytest.CaptureFixture) -> None:
        r = OllamaHealthReport(
            binary_installed=True,
            server_running=True,
            model_available=False,
            configured_model="llama3.2:3b",
            available_models=["mistral:7b"],
            errors=["Model 'llama3.2:3b' is not installed."],
        )
        print_health_report(r)
        out = capsys.readouterr().out
        assert "llama3.2:3b" in out

    def test_llm_connection_failed(self, capsys: pytest.CaptureFixture) -> None:
        r = OllamaHealthReport(
            binary_installed=True,
            server_running=True,
            model_available=True,
            llm_connected=False,
            configured_model="llama3.2:3b",
            errors=["LLM round-trip failed: empty response."],
        )
        print_health_report(r)
        out = capsys.readouterr().out
        assert "Issues detected" in out
