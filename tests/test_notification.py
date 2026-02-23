"""Tests for notification tools."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestSendEmail:
    def test_dry_run(self, tmp_path):
        with patch("ct.agent.config.Config.load") as mock_load:
            config_inst = MagicMock()
            config_inst.get.side_effect = lambda k, d=None: {
                "notification.from_email": "test@celltype.bio",
                "notification.sendgrid_api_key": None,
            }.get(k, d)
            mock_load.return_value = config_inst

            with patch("pathlib.Path.home", return_value=tmp_path):
                from ct.tools.notification import send_email
                result = send_email(
                    to="user@example.com",
                    subject="Test Subject",
                    body="Test body content",
                    dry_run=True,
                )

        assert "summary" in result
        assert result["dry_run"] is True
        assert "DRY RUN" in result["summary"]
        assert result["to"] == "user@example.com"
        assert result["subject"] == "Test Subject"
        assert result["body"] == "Test body content"

    def test_dry_run_logs(self, tmp_path):
        with patch("ct.agent.config.Config.load") as mock_load:
            config_inst = MagicMock()
            config_inst.get.side_effect = lambda k, d=None: d
            mock_load.return_value = config_inst

            with patch("pathlib.Path.home", return_value=tmp_path):
                from ct.tools.notification import send_email
                send_email(
                    to="user@example.com",
                    subject="Test",
                    body="Test body",
                    dry_run=True,
                )

            log_file = tmp_path / ".ct" / "sent_emails.log"
            assert log_file.exists()
            content = log_file.read_text()
            assert "DRY_RUN" in content
            assert "user@example.com" in content

    def test_missing_api_key(self, tmp_path):
        with patch("ct.agent.config.Config.load") as mock_load:
            config_inst = MagicMock()
            config_inst.get.side_effect = lambda k, d=None: None
            mock_load.return_value = config_inst

            with patch("pathlib.Path.home", return_value=tmp_path):
                from ct.tools.notification import send_email
                result = send_email(
                    to="user@example.com",
                    subject="Test",
                    body="Test body",
                    dry_run=False,
                )

        assert result["sent"] is False
        assert "api_key" in result.get("error", "")

    @patch("httpx.post")
    def test_send_actual(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch("ct.agent.config.Config.load") as mock_load:
            config_inst = MagicMock()
            config_inst.get.side_effect = lambda k, d=None: {
                "notification.sendgrid_api_key": "SG.test-key",
                "notification.from_email": "ct@celltype.bio",
            }.get(k, d)
            mock_load.return_value = config_inst

            with patch("pathlib.Path.home", return_value=tmp_path):
                from ct.tools.notification import send_email
                result = send_email(
                    to="user@example.com",
                    subject="Test",
                    body="Test body",
                    dry_run=False,
                )

        assert result["sent"] is True
        assert "summary" in result
        mock_post.assert_called_once()

    def test_custom_from_email(self, tmp_path):
        with patch("ct.agent.config.Config.load") as mock_load:
            config_inst = MagicMock()
            config_inst.get.side_effect = lambda k, d=None: d
            mock_load.return_value = config_inst

            with patch("pathlib.Path.home", return_value=tmp_path):
                from ct.tools.notification import send_email
                result = send_email(
                    to="user@example.com",
                    subject="Test",
                    body="Body",
                    from_email="custom@example.com",
                    dry_run=True,
                )

        assert result["from_email"] == "custom@example.com"
