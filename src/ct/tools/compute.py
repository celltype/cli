"""
GPU compute tools: list providers, estimate costs, submit jobs, check status.

PLACEHOLDER IMPLEMENTATION: Provider listings and pricing come from a static JSON
file bundled with ct. Prices may be outdated. submit_job and job_status make real
API calls when dry_run=False, but list_providers and estimate_cost use static data.
A real implementation would query live provider APIs for current pricing.
"""

import json
import time
from pathlib import Path
from ct.tools import registry
from ct.tools.http_client import request_json


# Module-level cache for provider data
_providers_data = None


def _load_providers() -> dict:
    """Load provider data from JSON, caching after first read."""
    global _providers_data
    if _providers_data is None:
        json_path = Path(__file__).resolve().parent.parent / "data" / "compute_providers.json"
        with open(json_path) as f:
            _providers_data = json.load(f)
    return _providers_data


def _provider_info(provider_id: str) -> dict | None:
    """Return provider metadata by ID."""
    data = _load_providers()
    for provider in data.get("providers", []):
        if provider.get("id") == provider_id:
            return provider
    return None


def _provider_api_key_config(provider: str) -> str:
    """Map provider IDs to config keys, with a predictable fallback."""
    key_map = {"lambda": "compute.lambda_api_key", "runpod": "compute.runpod_api_key"}
    return key_map.get(provider, f"compute.{provider}_api_key")


def _resolve_provider(provider: str | None, cfg) -> str:
    """Resolve provider from argument or config default."""
    if provider:
        return provider
    resolved = cfg.get("compute.default_provider", "lambda")
    return resolved or "lambda"


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict,
    payload: dict | None = None,
    timeout: int = 30,
    retries: int = 2,
) -> tuple[dict | None, str | None]:
    """HTTP request with shared retry/backoff semantics."""
    return request_json(
        method,
        url,
        json=payload,
        headers=headers,
        timeout=timeout,
        retries=retries,
    )


def _find_cheapest_gpu(min_vram_gb: int, provider_id: str = None) -> dict | None:
    """Find the cheapest GPU meeting VRAM requirements, optionally filtered by provider."""
    data = _load_providers()
    candidates = []
    for provider in data["providers"]:
        if provider_id and provider["id"] != provider_id:
            continue
        for gpu in provider["gpu_types"]:
            if gpu["vram_gb"] >= min_vram_gb:
                candidates.append({
                    "provider_id": provider["id"],
                    "provider_name": provider["name"],
                    "gpu_id": gpu["id"],
                    "gpu_name": gpu["name"],
                    "vram_gb": gpu["vram_gb"],
                    "price_per_hour": gpu["price_per_hour"],
                })
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["price_per_hour"])


@registry.register(
    name="compute.list_providers",
    description="List available GPU cloud providers with reference pricing for each GPU type (from built-in directory)",
    category="compute",
    parameters={},
    usage_guide="You need to see available GPU cloud providers and their pricing before submitting a compute job.",
)
def list_providers(**kwargs) -> dict:
    """List all GPU cloud providers and their available GPU types with pricing."""
    data = _load_providers()
    providers = []
    for p in data["providers"]:
        gpu_list = []
        for gpu in p["gpu_types"]:
            gpu_list.append({
                "id": gpu["id"],
                "name": gpu["name"],
                "vram_gb": gpu["vram_gb"],
                "price_per_hour": gpu["price_per_hour"],
            })
        providers.append({
            "id": p["id"],
            "name": p["name"],
            "website": p["website"],
            "gpu_types": gpu_list,
        })

    # Build summary
    lines = []
    for p in providers:
        gpu_strs = [f"{g['name']} (${g['price_per_hour']:.2f}/hr)" for g in p["gpu_types"]]
        lines.append(f"{p['name']}: {', '.join(gpu_strs)}")

    return {
        "summary": f"[PLACEHOLDER] {len(providers)} GPU cloud providers (static reference pricing — may be outdated):\n" + "\n".join(lines),
        "placeholder": True,
        "providers": providers,
    }


@registry.register(
    name="compute.estimate_cost",
    description="Estimate cost and time for a GPU computation job based on built-in templates (Boltz-2, AlphaFold, MD, virtual screening, training)",
    category="compute",
    parameters={
        "job_type": "Type of job: boltz2, alphafold, molecular_dynamics, virtual_screening, model_training",
        "n_samples": "Number of samples/structures to process (default 1)",
        "gpu_type": "Specific GPU type to use (optional, auto-selects cheapest if omitted)",
        "provider": "Provider ID: lambda or runpod (optional, auto-selects cheapest if omitted)",
    },
    usage_guide="You need to estimate the cost and time for a GPU computation (Boltz-2, AlphaFold, MD simulation, etc.) before deciding whether to proceed.",
)
def estimate_cost(job_type: str, n_samples: int = 1, gpu_type: str = None, provider: str = None, **kwargs) -> dict:
    """Estimate cost and time for a compute job.

    Looks up the job template, finds the best GPU option, and calculates
    estimated_hours and estimated_cost based on per-sample time and pricing.
    """
    data = _load_providers()
    templates = data["job_templates"]

    if job_type not in templates:
        valid = ", ".join(templates.keys())
        return {"error": f"Unknown job type '{job_type}'. Valid types: {valid}", "summary": f"Unknown job type '{job_type}'. Valid types: {valid}"}
    template = templates[job_type]
    min_vram = template["gpu_requirement_vram_gb"]

    # Resolve GPU selection
    if gpu_type and provider:
        # Find specific GPU at specific provider
        selected = None
        for p in data["providers"]:
            if p["id"] == provider:
                for g in p["gpu_types"]:
                    if g["id"] == gpu_type:
                        selected = {
                            "provider_id": p["id"],
                            "provider_name": p["name"],
                            "gpu_id": g["id"],
                            "gpu_name": g["name"],
                            "vram_gb": g["vram_gb"],
                            "price_per_hour": g["price_per_hour"],
                        }
                        break
        if not selected:
            return {"error": f"GPU '{gpu_type}' not found at provider '{provider}'", "summary": f"GPU '{gpu_type}' not found at provider '{provider}'"}
        if selected["vram_gb"] < min_vram:
            return {"error": f"GPU {gpu_type} has {selected['vram_gb']}GB VRAM but {job_type} requires {min_vram}GB", "summary": f"GPU {gpu_type} has {selected['vram_gb']}GB VRAM but {job_type} requires {min_vram}GB"}
    else:
        selected = _find_cheapest_gpu(min_vram, provider_id=provider)
        if not selected:
            msg = f"No GPU found with >= {min_vram}GB VRAM" + (f" at provider '{provider}'" if provider else "")
            return {"error": msg, "summary": msg}

    # Calculate cost
    time_per_sample = template["estimated_time_per_sample_minutes"]
    total_minutes = time_per_sample * n_samples
    total_hours = total_minutes / 60.0
    estimated_cost = total_hours * selected["price_per_hour"]

    return {
        "summary": (
            f"[PLACEHOLDER] {template['description']}: {n_samples} sample(s) on {selected['gpu_name']} ({selected['provider_name']})\n"
            f"Estimated time: {total_hours:.1f} hours | Estimated cost: ${estimated_cost:.2f} (based on static reference pricing — may be outdated)"
        ),
        "placeholder": True,
        "job_type": job_type,
        "n_samples": n_samples,
        "estimated_hours": round(total_hours, 2),
        "estimated_cost": round(estimated_cost, 2),
        "gpu": selected["gpu_id"],
        "gpu_name": selected["gpu_name"],
        "vram_gb": selected["vram_gb"],
        "provider": selected["provider_id"],
        "provider_name": selected["provider_name"],
        "price_per_hour": selected["price_per_hour"],
        "time_per_sample_minutes": time_per_sample,
    }


@registry.register(
    name="compute.submit_job",
    description="Submit a GPU compute job to a cloud provider (dry_run=True by default)",
    category="compute",
    parameters={
        "job_type": "Type of job: boltz2, alphafold, molecular_dynamics, virtual_screening, model_training",
        "params": "Job-specific parameters (dict with input files, config, etc.)",
        "provider": "Provider ID: lambda or runpod (default: compute.default_provider)",
        "gpu_type": "GPU type to use (optional, auto-selects if omitted)",
        "dry_run": "If True (default), only show what would be submitted without actually submitting",
    },
    usage_guide="You want to submit a GPU computation job. Always runs in dry_run mode unless explicitly overridden. Use after compute.estimate_cost.",
)
def submit_job(job_type: str, params: dict = None, provider: str | None = None, gpu_type: str = None, dry_run: bool = True, **kwargs) -> dict:
    """Submit a compute job to a cloud GPU provider.

    By default runs in dry_run mode, returning what would be submitted.
    When dry_run=False, POSTs the job to the provider API.
    """
    params = params or {}
    from ct.agent.config import Config
    cfg = Config.load()
    provider = _resolve_provider(provider, cfg)

    data = _load_providers()
    templates = data["job_templates"]

    if job_type not in templates:
        valid = ", ".join(templates.keys())
        return {"error": f"Unknown job type '{job_type}'. Valid types: {valid}", "summary": f"Unknown job type '{job_type}'. Valid types: {valid}"}
    template = templates[job_type]

    # Get cost estimate
    cost_est = estimate_cost(job_type=job_type, n_samples=params.get("n_samples", 1), gpu_type=gpu_type, provider=provider)
    if "error" in cost_est:
        return cost_est

    # Find provider API base
    provider_info = _provider_info(provider)
    if not provider_info:
        return {"error": f"Unknown provider '{provider}'", "summary": f"Unknown provider '{provider}'"}
    job_payload = {
        "job_type": job_type,
        "description": template["description"],
        "gpu_type": cost_est["gpu"],
        "provider": provider,
        "params": params,
        "estimated_hours": cost_est["estimated_hours"],
        "estimated_cost": cost_est["estimated_cost"],
    }

    if dry_run:
        return {
            "summary": (
                f"[DRY RUN] Would submit {template['description']} to {provider_info['name']}\n"
                f"GPU: {cost_est['gpu_name']} | Est. time: {cost_est['estimated_hours']:.1f}h | Est. cost: ${cost_est['estimated_cost']:.2f}\n"
                f"Set dry_run=False to actually submit."
            ),
            "dry_run": True,
            "job_payload": job_payload,
        }

    # Actual submission — validate API key first
    config_key = _provider_api_key_config(provider)
    api_key = cfg.get(config_key)
    if not api_key:
        return {
            "error": (
                f"No API key configured for {provider_info['name']}. "
                f"Set it with: ct config set {config_key} <your-key>\n"
                f"Or: export {config_key.upper().replace('.', '_')}=<your-key>\n"
                f"Sign up at: {provider_info.get('website', 'N/A')}\n"
                f"Run 'ct keys' to see all API key status."
            ),
            "summary": f"Job submission failed: no API key for {provider_info['name']}",
        }

    api_base = provider_info.get("api_base_url")
    if not api_base:
        return {
            "error": f"Provider '{provider}' does not have an API endpoint configured",
            "summary": f"Job submission failed: provider '{provider_info.get('name', provider)}' is not API-enabled",
        }

    api_url = f"{api_base}/jobs"
    result, request_error = _request_json(
        "POST",
        api_url,
        payload=job_payload,
        timeout=30,
        retries=2,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if request_error:
        return {
            "error": f"Job submission failed: {request_error}",
            "summary": f"Job submission failed for {provider_info['name']}: {request_error}",
        }

    job_id = result.get("id") or result.get("job_id") or "unknown"

    return {
        "summary": (
            f"Job submitted to {provider_info['name']}: {job_id}\n"
            f"GPU: {cost_est['gpu_name']} | Est. time: {cost_est['estimated_hours']:.1f}h | Est. cost: ${cost_est['estimated_cost']:.2f}"
        ),
        "job_id": job_id,
        "provider": provider,
        "job_payload": job_payload,
        "raw_response": result,
    }


@registry.register(
    name="compute.job_status",
    description="Check the status of a previously submitted compute job",
    category="compute",
    parameters={
        "job_id": "Job ID returned from compute.submit_job",
        "provider": "Provider ID: lambda or runpod (default: compute.default_provider)",
    },
    usage_guide="You want to check the status of a previously submitted compute job.",
)
def job_status(job_id: str, provider: str | None = None, **kwargs) -> dict:
    """Check the status of a compute job via the provider API."""
    from ct.agent.config import Config
    cfg = Config.load()
    provider = _resolve_provider(provider, cfg)

    provider_info = _provider_info(provider)
    if not provider_info:
        return {"error": f"Unknown provider '{provider}'", "summary": f"Unknown provider '{provider}'"}
    config_key = _provider_api_key_config(provider)
    api_key = cfg.get(config_key)
    if not api_key:
        return {
            "error": (
                f"No API key configured for {provider_info['name']}. "
                f"Set it with: ct config set {config_key} <your-key>\n"
                f"Run 'ct keys' to see all API key status."
            ),
            "summary": f"Job status check failed: no API key for {provider_info['name']}",
        }

    api_base = provider_info.get("api_base_url")
    if not api_base:
        return {
            "error": f"Provider '{provider}' does not have an API endpoint configured",
            "summary": f"Job status check failed: provider '{provider_info.get('name', provider)}' is not API-enabled",
        }

    api_url = f"{api_base}/jobs/{job_id}"
    result, request_error = _request_json(
        "GET",
        api_url,
        timeout=30,
        retries=2,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if request_error:
        return {
            "error": f"Failed to check job status: {request_error}",
            "summary": f"Job status check failed for {provider_info['name']}: {request_error}",
        }

    status = result.get("status", "unknown")
    progress = result.get("progress", None)
    elapsed = result.get("elapsed_seconds", None)

    elapsed_str = ""
    if elapsed is not None:
        hours = elapsed / 3600
        elapsed_str = f" | Elapsed: {hours:.1f}h"

    progress_str = ""
    if progress is not None:
        progress_str = f" | Progress: {progress}%"

    return {
        "summary": f"Job {job_id} ({provider_info['name']}): {status}{progress_str}{elapsed_str}",
        "job_id": job_id,
        "provider": provider,
        "status": status,
        "progress": progress,
        "elapsed_seconds": elapsed,
        "raw_response": result,
    }
