"""Tests for shared HTTP client helpers used by tool modules."""

from unittest.mock import MagicMock, patch

from ct.tools.http_client import request, request_json


class TestHttpClient:
    @patch("httpx.get")
    def test_request_json_success(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"ok": True}
        mock_get.return_value = resp

        data, error = request_json("GET", "https://example.org/api")

        assert error is None
        assert data == {"ok": True}
        assert mock_get.call_count == 1

    @patch("ct.tools.http_client.time.sleep", return_value=None)
    @patch("httpx.get")
    def test_request_json_retries_on_503(self, mock_get, _sleep):
        transient = MagicMock()
        transient.status_code = 503
        transient.raise_for_status.return_value = None

        ok = MagicMock()
        ok.status_code = 200
        ok.raise_for_status.return_value = None
        ok.json.return_value = {"ok": True}

        mock_get.side_effect = [transient, ok]

        data, error = request_json("GET", "https://example.org/api", retries=2)

        assert error is None
        assert data == {"ok": True}
        assert mock_get.call_count == 2

    @patch("httpx.get")
    def test_request_returns_http_error_message(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "Not found"
        resp.raise_for_status.side_effect = Exception("boom")
        mock_get.return_value = resp

        _resp, error = request("GET", "https://example.org/missing")

        assert error is not None
        assert "boom" in error or "HTTP 404" in error

    @patch("httpx.get")
    def test_request_can_skip_raise_for_status(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "missing"
        mock_get.return_value = resp

        got_resp, error = request(
            "GET",
            "https://example.org/missing",
            raise_for_status=False,
            retries=0,
        )

        assert error is None
        assert got_resp.status_code == 404

    @patch("httpx.get")
    def test_request_json_invalid_json(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("invalid")
        mock_get.return_value = resp

        data, error = request_json("GET", "https://example.org/api")

        assert data is None
        assert "Invalid JSON response" in error

    @patch("httpx.get")
    def test_request_get_does_not_forward_json_or_data_kwargs(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        mock_get.return_value = resp

        _resp, error = request(
            "GET",
            "https://example.org/api",
            json={"a": 1},
            data={"b": 2},
            retries=0,
        )

        assert error is None
        _, kwargs = mock_get.call_args
        assert "json" not in kwargs
        assert "data" not in kwargs
