"""Tests for doctor health checks."""

import time
from unittest.mock import MagicMock, patch

from ct.agent.config import Config
from ct.agent.doctor import (
    DoctorCheck,
    _check_api_connectivity,
    _check_data_availability,
    _check_downloads_dir,
    _check_tool_health,
    run_checks,
)


def _checks_by_name(checks):
    return {c.name: c for c in checks}


def test_doctor_accepts_pharma_profile_and_style():
    cfg = Config(
        data={
            "llm.provider": "anthropic",
            "llm.api_key": "x",
            "agent.profile": "pharma",
            "agent.synthesis_style": "pharma",
        }
    )
    checks = _checks_by_name(run_checks(cfg))
    assert checks["runtime_profile"].status == "ok"
    assert checks["runtime_profile"].detail == "pharma"
    assert checks["synthesis_style"].status == "ok"
    assert checks["synthesis_style"].detail == "pharma"


def test_doctor_warns_for_unknown_synthesis_style():
    cfg = Config(
        data={
            "llm.provider": "anthropic",
            "llm.api_key": "x",
            "agent.profile": "research",
            "agent.synthesis_style": "mystery",
        }
    )
    checks = _checks_by_name(run_checks(cfg))
    assert checks["runtime_profile"].status == "ok"
    assert checks["synthesis_style"].status == "warn"
    assert "Unknown agent.synthesis_style" in checks["synthesis_style"].detail


def test_doctor_warns_when_pharma_profile_not_using_pharma_style():
    cfg = Config(
        data={
            "llm.provider": "anthropic",
            "llm.api_key": "x",
            "agent.profile": "pharma",
            "agent.synthesis_style": "standard",
        }
    )
    checks = _checks_by_name(run_checks(cfg))
    assert checks["runtime_profile"].status == "ok"
    assert checks["synthesis_style"].status == "warn"
    assert "agent.profile=pharma" in checks["synthesis_style"].detail


# ---------------------------------------------------------------------------
# Runtime health checks (Part A)
# ---------------------------------------------------------------------------


class TestDataAvailability:
    @patch("ct.agent.doctor.Path.home")
    def test_all_datasets_missing(self, mock_home, tmp_path):
        mock_home.return_value = tmp_path / "fakehome"
        cfg = Config(data={"data.base": str(tmp_path / "empty_data")})
        check = _check_data_availability(cfg)
        assert check.name == "data_availability"
        assert check.status == "warn"
        assert "depmap" in check.detail
        assert "prism" in check.detail
        assert "l1000" in check.detail

    @patch("ct.agent.doctor.Path.home")
    def test_some_datasets_found(self, mock_home, tmp_path):
        mock_home.return_value = tmp_path / "fakehome"
        data_dir = tmp_path / "data" / "depmap"
        data_dir.mkdir(parents=True)
        (data_dir / "CRISPRGeneEffect.csv").write_text("header\n")
        cfg = Config(data={"data.base": str(tmp_path / "data")})
        check = _check_data_availability(cfg)
        assert check.status == "warn"
        assert "depmap" in check.detail  # found
        assert "prism" in check.detail   # missing

    def test_all_datasets_found(self, tmp_path):
        data_dir = tmp_path / "data"
        for sub, fname in [
            ("depmap", "CRISPRGeneEffect.csv"),
            ("prism", "prism_LFC_COLLAPSED.csv"),
            ("l1000", "l1000_landmark_only.parquet"),
        ]:
            d = data_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / fname).write_text("header\n")
        cfg = Config(data={"data.base": str(data_dir)})
        check = _check_data_availability(cfg)
        assert check.status == "ok"
        assert "depmap" in check.detail


class TestDownloadsDir:
    def test_downloads_dir_ok(self):
        check = _check_downloads_dir()
        assert check.name == "downloads_dir"
        # On a normal system this should succeed
        assert check.status == "ok"


class TestApiConnectivity:
    def test_all_reachable(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.head", return_value=mock_resp):
            checks = _check_api_connectivity()
        by_name = {c.name: c for c in checks}
        assert by_name["api_connectivity"].status == "ok"

    def test_some_unreachable(self):
        def mock_head(url, **kwargs):
            if "eutils" in url:
                raise ConnectionError("no route")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch("httpx.head", side_effect=mock_head):
            checks = _check_api_connectivity()
        by_name = {c.name: c for c in checks}
        assert by_name["api_connectivity"].status == "warn"
        assert "PubMed eutils" in by_name["api_connectivity"].detail

    def test_httpx_not_installed(self):
        with patch.dict("sys.modules", {"httpx": None}):
            # Force ImportError in the function
            import importlib
            import ct.agent.doctor
            # The function catches ImportError internally
            checks = _check_api_connectivity()
            # If httpx is actually installed we just verify the check ran
            assert len(checks) >= 1

    def test_all_unreachable(self):
        with patch("httpx.head", side_effect=ConnectionError("offline")):
            checks = _check_api_connectivity()
        by_name = {c.name: c for c in checks}
        assert by_name["api_connectivity"].status == "warn"
        assert "All API probes failed" in by_name["api_connectivity"].detail


class TestToolHealth:
    def test_no_failures(self):
        session = MagicMock()
        session.tool_health_suppressed_tools.return_value = set()
        session._tool_health_failures = {}
        check = _check_tool_health(session)
        assert check.name == "tool_health"
        assert check.status == "ok"

    def test_with_suppressed_tools(self):
        session = MagicMock()
        session.tool_health_suppressed_tools.return_value = {"literature.pubmed_search"}
        session._tool_health_failures = {
            "literature.pubmed_search": [time.time(), time.time()],
        }
        check = _check_tool_health(session)
        assert check.status == "warn"
        assert "Suppressed" in check.detail
        assert "pubmed_search" in check.detail

    def test_with_failures_not_yet_suppressed(self):
        session = MagicMock()
        session.tool_health_suppressed_tools.return_value = set()
        session._tool_health_failures = {
            "data_api.uniprot_lookup": [time.time()],
        }
        check = _check_tool_health(session)
        assert check.status == "warn"
        assert "Recent failures" in check.detail
        assert "uniprot_lookup" in check.detail


class TestDoctorWithSession:
    def test_run_checks_includes_tool_health_when_session_provided(self):
        cfg = Config(data={"llm.provider": "anthropic", "llm.api_key": "x"})
        session = MagicMock()
        session.tool_health_suppressed_tools.return_value = set()
        session._tool_health_failures = {}
        with patch("ct.agent.doctor._check_api_connectivity", return_value=[
            DoctorCheck(name="api_connectivity", status="ok", detail="mocked")
        ]):
            checks = _checks_by_name(run_checks(cfg, session=session))
        assert "tool_health" in checks
        assert checks["tool_health"].status == "ok"

    def test_run_checks_reports_tool_health_unavailable_without_session(self):
        cfg = Config(data={"llm.provider": "anthropic", "llm.api_key": "x"})
        with patch("ct.agent.doctor._check_api_connectivity", return_value=[
            DoctorCheck(name="api_connectivity", status="ok", detail="mocked")
        ]):
            checks = _checks_by_name(run_checks(cfg))
        assert "tool_health" in checks
        assert checks["tool_health"].status == "warn"
        assert "No active session context" in checks["tool_health"].detail


class TestPreflightValidationConfig:
    def test_preflight_enabled_by_default(self):
        cfg = Config(data={"llm.provider": "anthropic", "llm.api_key": "x"})
        with patch("ct.agent.doctor._check_api_connectivity", return_value=[
            DoctorCheck(name="api_connectivity", status="ok", detail="mocked")
        ]):
            checks = _checks_by_name(run_checks(cfg))
        assert checks["preflight_validation"].status == "ok"

    def test_preflight_disabled_warns(self):
        cfg = Config(data={
            "llm.provider": "anthropic",
            "llm.api_key": "x",
            "agent.preflight_validation_enabled": False,
        })
        with patch("ct.agent.doctor._check_api_connectivity", return_value=[
            DoctorCheck(name="api_connectivity", status="ok", detail="mocked")
        ]):
            checks = _checks_by_name(run_checks(cfg))
        assert checks["preflight_validation"].status == "warn"
