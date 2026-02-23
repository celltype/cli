"""Tests for compute tools."""

import pytest
from unittest.mock import patch, MagicMock

MOCK_PROVIDERS = {
    "providers": [
        {
            "id": "lambda",
            "name": "Lambda Labs",
            "website": "https://lambdalabs.com",
            "api_base_url": "https://cloud.lambdalabs.com/api/v1",
            "gpu_types": [
                {"id": "A100_80GB", "name": "NVIDIA A100 80GB", "vram_gb": 80, "price_per_hour": 1.29},
                {"id": "H100_80GB", "name": "NVIDIA H100 80GB", "vram_gb": 80, "price_per_hour": 2.49},
                {"id": "A10_24GB", "name": "NVIDIA A10 24GB", "vram_gb": 24, "price_per_hour": 0.75},
            ],
        },
        {
            "id": "runpod",
            "name": "RunPod",
            "website": "https://runpod.io",
            "api_base_url": "https://api.runpod.ai/v2",
            "gpu_types": [
                {"id": "A100_80GB", "name": "NVIDIA A100 80GB", "vram_gb": 80, "price_per_hour": 1.19},
                {"id": "RTX4090_24GB", "name": "NVIDIA RTX 4090 24GB", "vram_gb": 24, "price_per_hour": 0.44},
            ],
        },
    ],
    "job_templates": {
        "boltz2": {
            "description": "Boltz-2 structure prediction",
            "gpu_requirement_vram_gb": 40,
            "estimated_time_per_sample_minutes": 15,
            "recommended_gpu": "A100_80GB",
        },
        "alphafold": {
            "description": "AlphaFold structure prediction",
            "gpu_requirement_vram_gb": 16,
            "estimated_time_per_sample_minutes": 30,
            "recommended_gpu": "A100_80GB",
        },
        "model_training": {
            "description": "Model training",
            "gpu_requirement_vram_gb": 24,
            "estimated_time_per_sample_minutes": 60,
            "recommended_gpu": "A100_80GB",
        },
    },
}


@pytest.fixture(autouse=True)
def mock_providers():
    with patch("ct.tools.compute._load_providers", return_value=MOCK_PROVIDERS):
        import ct.tools.compute as compute_mod
        compute_mod._providers_data = None
        yield


@pytest.fixture(autouse=True)
def mock_compute_config():
    class _Cfg:
        _values = {
            "compute.lambda_api_key": "test-lambda-key",
            "compute.runpod_api_key": "test-runpod-key",
            "compute.default_provider": "lambda",
        }

        def get(self, key, default=None):
            return self._values.get(key, default)

    with patch("ct.agent.config.Config.load", return_value=_Cfg()):
        yield


class TestListProviders:
    def test_lists_all_providers(self):
        from ct.tools.compute import list_providers
        result = list_providers()
        assert "summary" in result
        assert len(result["providers"]) == 2

    def test_has_gpu_info(self):
        from ct.tools.compute import list_providers
        result = list_providers()
        lambda_provider = [p for p in result["providers"] if p["id"] == "lambda"][0]
        assert len(lambda_provider["gpu_types"]) == 3
        gpu = lambda_provider["gpu_types"][0]
        assert "vram_gb" in gpu
        assert "price_per_hour" in gpu


class TestEstimateCost:
    def test_basic_estimate(self):
        from ct.tools.compute import estimate_cost
        result = estimate_cost(job_type="boltz2", n_samples=1)
        assert "summary" in result
        assert result["estimated_hours"] > 0
        assert result["estimated_cost"] > 0
        # Should pick cheapest GPU with >= 40GB VRAM
        assert result["vram_gb"] >= 40

    def test_scaling_with_samples(self):
        from ct.tools.compute import estimate_cost
        result_1 = estimate_cost(job_type="boltz2", n_samples=1)
        result_50 = estimate_cost(job_type="boltz2", n_samples=50)
        assert result_50["estimated_cost"] == pytest.approx(result_1["estimated_cost"] * 50, rel=0.01)
        assert result_50["estimated_hours"] == pytest.approx(result_1["estimated_hours"] * 50, rel=0.01)

    def test_specific_provider(self):
        from ct.tools.compute import estimate_cost
        result = estimate_cost(job_type="boltz2", provider="lambda")
        assert result["provider"] == "lambda"

    def test_specific_gpu_and_provider(self):
        from ct.tools.compute import estimate_cost
        result = estimate_cost(job_type="boltz2", gpu_type="H100_80GB", provider="lambda")
        assert result["gpu"] == "H100_80GB"
        assert result["price_per_hour"] == 2.49

    def test_unknown_job_type(self):
        from ct.tools.compute import estimate_cost
        result = estimate_cost(job_type="nonexistent")
        assert "error" in result

    def test_insufficient_vram(self):
        from ct.tools.compute import estimate_cost
        # boltz2 requires 40GB, RTX4090 only has 24GB
        result = estimate_cost(job_type="boltz2", gpu_type="RTX4090_24GB", provider="runpod")
        assert "error" in result

    def test_selects_cheapest_gpu(self):
        from ct.tools.compute import estimate_cost
        # For boltz2 (40GB req), cheapest is RunPod A100 at $1.19/hr
        result = estimate_cost(job_type="boltz2")
        assert result["price_per_hour"] == 1.19
        assert result["provider"] == "runpod"


class TestSubmitJob:
    def test_submit_dry_run(self):
        from ct.tools.compute import submit_job
        result = submit_job(
            job_type="boltz2",
            params={"n_samples": 5, "input_file": "complexes.csv"},
            dry_run=True,
        )
        assert "summary" in result
        assert result["dry_run"] is True
        assert "DRY RUN" in result["summary"]
        assert "job_payload" in result

    def test_submit_unknown_job(self):
        from ct.tools.compute import submit_job
        result = submit_job(job_type="nonexistent", dry_run=True)
        assert "error" in result

    @patch("httpx.post")
    def test_submit_actual(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "job-12345", "status": "queued"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        from ct.tools.compute import submit_job
        result = submit_job(
            job_type="boltz2",
            params={"n_samples": 1},
            provider="lambda",
            dry_run=False,
        )
        assert result["job_id"] == "job-12345"
        assert "summary" in result

    @patch("ct.tools.compute.time.sleep", return_value=None)
    @patch("httpx.post")
    def test_submit_retries_on_transient_http_status(self, mock_post, _sleep):
        transient = MagicMock()
        transient.status_code = 503
        transient.text = "Service unavailable"

        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"id": "job-67890", "status": "queued"}
        ok.raise_for_status.return_value = None

        mock_post.side_effect = [transient, ok]

        from ct.tools.compute import submit_job
        result = submit_job(
            job_type="boltz2",
            params={"n_samples": 1},
            provider="lambda",
            dry_run=False,
        )
        assert result["job_id"] == "job-67890"
        assert mock_post.call_count == 2

    @patch("httpx.post")
    def test_submit_uses_config_default_provider(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "job-default", "status": "queued"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        from ct.tools.compute import submit_job
        result = submit_job(
            job_type="boltz2",
            params={"n_samples": 1},
            provider=None,
            dry_run=False,
        )
        assert result["provider"] == "lambda"
        assert result["job_id"] == "job-default"


class TestJobStatus:
    @patch("httpx.get")
    def test_job_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "running",
            "progress": 45,
            "elapsed_seconds": 3600,
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from ct.tools.compute import job_status
        result = job_status(job_id="job-12345", provider="lambda")
        assert "summary" in result
        assert result["status"] == "running"
        assert result["progress"] == 45

    @patch("httpx.get")
    def test_job_status_unknown_provider(self, mock_get):
        from ct.tools.compute import job_status
        result = job_status(job_id="job-12345", provider="nonexistent")
        assert "error" in result
