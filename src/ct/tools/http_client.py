"""
Shared HTTP helpers for ct tools.

Provides retry/backoff, normalized errors, and JSON parsing wrappers for
API-heavy tool modules.
"""

import time


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _call_httpx(method: str, url: str, **kwargs):
    import httpx

    method = method.upper()
    # Avoid passing unsupported kwargs (e.g., json/data to httpx.get).
    cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if method == "GET":
        cleaned_kwargs.pop("json", None)
        cleaned_kwargs.pop("data", None)
        return httpx.get(url, **cleaned_kwargs)
    if method == "POST":
        return httpx.post(url, **cleaned_kwargs)
    return httpx.request(method, url, **cleaned_kwargs)


def _format_http_error(response) -> str:
    status = getattr(response, "status_code", "unknown")
    body = (getattr(response, "text", "") or "").strip().replace("\n", " ")
    body = body[:300]
    return f"HTTP {status}" + (f": {body}" if body else "")


def request(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    data: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 2,
    backoff_seconds: float = 0.5,
    raise_for_status: bool = True,
) -> tuple[object | None, str | None]:
    """Perform HTTP request with retry/backoff.

    Returns `(response, error)`. Exactly one is non-None.
    """
    try:
        import httpx
    except ImportError:
        return None, "httpx required (pip install httpx)"

    delay = max(backoff_seconds, 0.0)
    last_error = None

    for attempt in range(max(retries, 0) + 1):
        try:
            resp = _call_httpx(
                method,
                url,
                params=params,
                json=json,
                data=data,
                headers=headers,
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            return None, last_error
        except Exception as exc:
            return None, str(exc)

        status = int(getattr(resp, "status_code", 0) or 0)
        if status in _RETRYABLE_STATUS and attempt < retries:
            time.sleep(delay)
            delay *= 2
            continue

        if raise_for_status:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                return None, _format_http_error(resp)
            except Exception as exc:
                return None, str(exc)

        return resp, None

    return None, last_error or "Request failed"


def request_json(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    data: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 2,
    backoff_seconds: float = 0.5,
    raise_for_status: bool = True,
) -> tuple[dict | list | None, str | None]:
    """Perform HTTP request and parse JSON body."""
    resp, error = request(
        method,
        url,
        params=params,
        json=json,
        data=data,
        headers=headers,
        timeout=timeout,
        retries=retries,
        backoff_seconds=backoff_seconds,
        raise_for_status=raise_for_status,
    )
    if error:
        return None, error

    # Validate Content-Type before parsing — some APIs return HTML on 200
    content_type = ""
    try:
        ct_raw = resp.headers.get("content-type", "")
        if isinstance(ct_raw, str):
            content_type = ct_raw.lower()
    except Exception:
        pass
    if content_type and "json" not in content_type and "javascript" not in content_type:
        status = getattr(resp, "status_code", "unknown")
        return None, f"Expected JSON but got {content_type} (HTTP {status})"

    try:
        return resp.json(), None
    except Exception:
        status = getattr(resp, "status_code", "unknown")
        return None, f"Invalid JSON response (HTTP {status})"
