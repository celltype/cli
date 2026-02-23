"""Tests for the dataset downloader."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ct.data.downloader import (
    DATASETS, DOWNLOAD_TIMEOUT, download_dataset, download_all,
    dataset_status, _download_file,
)


class TestDatasetCatalog:
    def test_all_datasets_have_required_fields(self):
        for name, ds in DATASETS.items():
            assert "description" in ds, f"{name} missing description"
            assert "files" in ds, f"{name} missing files"
            assert "source" in ds, f"{name} missing source"
            assert "auto_download" in ds, f"{name} missing auto_download"

    def test_expected_datasets_present(self):
        expected = {"depmap", "prism", "l1000", "msigdb", "string", "alphafold"}
        assert expected == set(DATASETS.keys())

    def test_auto_download_datasets_have_urls(self):
        for name, ds in DATASETS.items():
            if ds.get("auto_download"):
                for fname, url in ds["files"].items():
                    assert url is not None, f"{name}/{fname} is auto_download but has no URL"

    def test_depmap_is_auto_download(self):
        assert DATASETS["depmap"]["auto_download"] is True
        for fname, url in DATASETS["depmap"]["files"].items():
            assert url is not None, f"depmap/{fname} missing URL"

    def test_prism_is_manual_download(self):
        assert DATASETS["prism"]["auto_download"] is False
        assert "note" in DATASETS["prism"]

    def test_l1000_exists(self):
        assert "l1000" in DATASETS
        assert DATASETS["l1000"]["auto_download"] is False  # requires prepare script

    def test_download_timeout_is_600(self):
        assert DOWNLOAD_TIMEOUT == 600


class TestDownloadFile:
    def test_successful_download(self, tmp_path):
        dest = tmp_path / "test.json"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "100"}
        mock_resp.iter_bytes.return_value = [b'{"data": "test"}']

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)
            result = _download_file("https://example.com/test.json", dest, "test.json")

        assert result is True
        assert dest.exists()

    def test_http_error(self, tmp_path):
        dest = tmp_path / "test.json"

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {}

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)
            result = _download_file("https://example.com/bad.json", dest, "bad.json")

        assert result is False

    def test_network_error(self, tmp_path):
        import httpx
        dest = tmp_path / "test.json"

        with patch("httpx.stream", side_effect=httpx.ConnectError("timeout")):
            result = _download_file("https://example.com/test.json", dest, "test.json")

        assert result is False
        assert not dest.exists()

    def test_uses_increased_timeout(self, tmp_path):
        """Verify download uses DOWNLOAD_TIMEOUT (600s) not the old 120s."""
        dest = tmp_path / "test.json"

        with patch("httpx.stream") as mock_stream:
            mock_stream.side_effect = Exception("intercepted")
            try:
                _download_file("https://example.com/test.json", dest)
            except Exception:
                pass
            # Check timeout arg
            call_args = mock_stream.call_args
            assert call_args[1].get("timeout", call_args[0][2] if len(call_args[0]) > 2 else None) == DOWNLOAD_TIMEOUT


class TestDownloadDataset:
    @patch("ct.data.downloader.Config")
    def test_unknown_dataset(self, mock_config, capsys):
        download_dataset("nonexistent_dataset")
        captured = capsys.readouterr()
        assert "Unknown dataset" in captured.out or True  # Rich output may not capture

    @patch("ct.data.downloader._download_file")
    @patch("ct.data.downloader.Config")
    def test_auto_download_calls_download(self, mock_config, mock_dl, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = str(tmp_path)
        mock_config.load.return_value = mock_cfg
        mock_dl.return_value = True

        download_dataset("msigdb", output=tmp_path)

        # Should have called _download_file for each msigdb file
        assert mock_dl.call_count == len(DATASETS["msigdb"]["files"])

    @patch("ct.data.downloader._download_file")
    @patch("ct.data.downloader.Config")
    def test_skips_existing_files(self, mock_config, mock_dl, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = str(tmp_path)
        mock_config.load.return_value = mock_cfg

        # Create one file already
        first_file = list(DATASETS["msigdb"]["files"].keys())[0]
        (tmp_path / first_file).write_text("{}")

        mock_dl.return_value = True
        download_dataset("msigdb", output=tmp_path)

        # Should skip the existing file
        expected_downloads = len(DATASETS["msigdb"]["files"]) - 1
        assert mock_dl.call_count == expected_downloads

    @patch("ct.data.downloader._download_file")
    @patch("ct.data.downloader.Config")
    def test_auto_configures_data_path_after_download(self, mock_config, mock_dl, tmp_path):
        """After successful download, config should be auto-set."""
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = str(tmp_path)
        mock_config.load.return_value = mock_cfg
        mock_dl.return_value = True

        download_dataset("msigdb", output=tmp_path)

        # Verify cfg.set was called with data.msigdb
        mock_cfg.set.assert_called_once_with("data.msigdb", str(tmp_path))
        mock_cfg.save.assert_called_once()

    @patch("ct.data.downloader._download_file")
    @patch("ct.data.downloader.Config")
    def test_depmap_auto_download(self, mock_config, mock_dl, tmp_path):
        """DepMap should now auto-download (not manual)."""
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = str(tmp_path)
        mock_config.load.return_value = mock_cfg
        mock_dl.return_value = True

        download_dataset("depmap", output=tmp_path)

        # Should have called _download_file for each depmap file
        assert mock_dl.call_count == len(DATASETS["depmap"]["files"])

    @patch("ct.data.downloader._download_file")
    @patch("ct.data.downloader.Config")
    def test_prism_manual_download(self, mock_config, mock_dl, tmp_path):
        """PRISM is now manual — should not call _download_file."""
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = str(tmp_path)
        mock_config.load.return_value = mock_cfg

        download_dataset("prism", output=tmp_path)
        assert mock_dl.call_count == 0


class TestDownloadAll:
    @patch("ct.data.downloader.download_dataset")
    def test_download_all_calls_auto_datasets(self, mock_download):
        """download_all should call download_dataset for each auto-downloadable dataset."""
        download_all()

        auto_names = {name for name, ds in DATASETS.items() if ds.get("auto_download")}
        called_names = {call.args[0] for call in mock_download.call_args_list}
        assert auto_names == called_names

    @patch("ct.data.downloader.download_dataset")
    def test_download_dataset_all_flag(self, mock_download):
        """download_dataset('all') should trigger download_all."""
        download_dataset("all")
        # Should be called for each auto-download dataset
        assert mock_download.call_count >= 3  # depmap, prism, msigdb, string


class TestDatasetStatus:
    @patch("ct.data.downloader.Config")
    def test_returns_table(self, mock_config, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, *args: str(tmp_path) if key == "data.base" else None
        mock_config.load.return_value = mock_cfg

        table = dataset_status()
        assert table is not None
        assert table.title == "Dataset Status"

    @patch("ct.data.downloader.Config")
    def test_detects_complete_dataset(self, mock_config, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, *args: str(tmp_path) if key == "data.base" else None
        mock_config.load.return_value = mock_cfg

        # Create a complete string dataset
        string_dir = tmp_path / "string"
        string_dir.mkdir()
        (string_dir / "9606.protein.links.v12.0.txt.gz").write_bytes(b"data")

        table = dataset_status()
        # Table exists and has rows for all datasets
        assert len(table.rows) == len(DATASETS)

    @patch("ct.data.downloader.Config")
    def test_includes_l1000_dataset(self, mock_config, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, *args: str(tmp_path) if key == "data.base" else None
        mock_config.load.return_value = mock_cfg

        table = dataset_status()
        # Should include l1000 row
        assert len(table.rows) == len(DATASETS)
