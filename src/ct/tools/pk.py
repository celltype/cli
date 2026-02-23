"""Pharmacokinetics tools for quick noncompartmental analysis (NCA)."""

from __future__ import annotations

import math

import numpy as np

from ct.tools import registry


def _to_float_list(values: list | None, field_name: str) -> tuple[list[float] | None, str | None]:
    if values is None:
        return None, f"'{field_name}' is required"
    out: list[float] = []
    try:
        for value in values:
            out.append(float(value))
    except Exception:
        return None, f"'{field_name}' must be a list of numeric values"
    return out, None


def _safe_round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


@registry.register(
    name="pk.nca_basic",
    description="Run basic noncompartmental PK analysis (Cmax/Tmax/AUC/lambda_z/t1/2; CL if dose provided)",
    category="pk",
    parameters={
        "times": "Sampling times (e.g., [0, 0.5, 1, 2, 4, 8, 24])",
        "concentrations": "Observed concentrations aligned with times",
        "dose": "Optional administered dose for CL/F and Vz/F",
        "route": "Route type: 'iv' or 'extravascular' (default extravascular)",
        "n_terminal_points": "Number of terminal points for lambda_z fit (default 3)",
        "lloq": "Optional lower limit of quantification; values below are set to 0",
        "subject_id": "Optional subject identifier for report labeling",
    },
    usage_guide=(
        "Use when you have concentration-time data and need rapid PK triage metrics. "
        "Returns noncompartmental metrics plus terminal-phase fit diagnostics."
    ),
)
def nca_basic(
    times: list | None = None,
    concentrations: list | None = None,
    dose: float | None = None,
    route: str = "extravascular",
    n_terminal_points: int = 3,
    lloq: float | None = None,
    subject_id: str = "",
    **kwargs,
) -> dict:
    """Perform a basic NCA workflow from concentration-time observations."""
    del kwargs

    t_list, t_error = _to_float_list(times, "times")
    if t_error:
        return {"summary": t_error, "error": "invalid_times"}
    c_list, c_error = _to_float_list(concentrations, "concentrations")
    if c_error:
        return {"summary": c_error, "error": "invalid_concentrations"}

    assert t_list is not None and c_list is not None

    if len(t_list) != len(c_list):
        return {
            "summary": f"Length mismatch: {len(t_list)} times vs {len(c_list)} concentrations",
            "error": "length_mismatch",
        }
    if len(t_list) < 3:
        return {
            "summary": f"Need at least 3 observations for NCA, got {len(t_list)}",
            "error": "insufficient_points",
        }

    warnings: list[str] = []
    pairs = []
    for time_value, conc_value in zip(t_list, c_list):
        if not np.isfinite(time_value) or not np.isfinite(conc_value):
            continue
        pairs.append((float(time_value), float(conc_value)))
    if len(pairs) < 3:
        return {
            "summary": "Need at least 3 finite concentration-time points after filtering.",
            "error": "insufficient_finite_points",
        }

    pairs.sort(key=lambda x: x[0])
    if pairs[0][0] < 0:
        return {"summary": "Negative sampling times are not allowed.", "error": "negative_time"}

    # Handle duplicate time points by averaging concentrations at each time.
    dedup: dict[float, list[float]] = {}
    for t_val, c_val in pairs:
        dedup.setdefault(t_val, []).append(c_val)
    if len(dedup) < len(pairs):
        warnings.append("Duplicate time points detected; concentrations were averaged per unique time.")

    times_sorted = np.array(sorted(dedup.keys()), dtype=float)
    conc_sorted = np.array([float(np.mean(dedup[t])) for t in times_sorted], dtype=float)
    if len(times_sorted) < 3:
        return {
            "summary": "Need at least 3 unique time points after deduplication.",
            "error": "insufficient_unique_times",
        }

    if lloq is not None:
        try:
            lloq_value = float(lloq)
        except Exception:
            return {"summary": "lloq must be numeric.", "error": "invalid_lloq"}
        if lloq_value < 0:
            return {"summary": "lloq cannot be negative.", "error": "invalid_lloq"}
        below = int(np.sum(conc_sorted < lloq_value))
        if below > 0:
            warnings.append(f"{below} concentration values below LLOQ were set to 0.")
            conc_sorted = np.where(conc_sorted < lloq_value, 0.0, conc_sorted)

    cmax_idx = int(np.argmax(conc_sorted))
    cmax = float(conc_sorted[cmax_idx])
    tmax = float(times_sorted[cmax_idx])

    auc_last = 0.0
    for i in range(1, len(times_sorted)):
        dt = float(times_sorted[i] - times_sorted[i - 1])
        if dt <= 0:
            return {
                "summary": "Sampling times must be strictly increasing after deduplication.",
                "error": "non_increasing_time",
            }
        auc_last += 0.5 * float(conc_sorted[i] + conc_sorted[i - 1]) * dt

    # Terminal elimination estimate (lambda_z) from last positive points.
    n_terminal_points = int(n_terminal_points or 3)
    if n_terminal_points < 3:
        n_terminal_points = 3
        warnings.append("n_terminal_points < 3 is not robust; using 3.")

    positive_idx = np.where(conc_sorted > 0)[0]
    lambda_z = None
    terminal_r2 = None
    half_life = None
    terminal_points_used = 0
    auc_extrap = None
    auc_inf = None
    extrapolated_fraction = None

    if len(positive_idx) >= 3:
        choose = min(n_terminal_points, len(positive_idx))
        terminal_slice = positive_idx[-choose:]
        t_term = times_sorted[terminal_slice]
        c_term = conc_sorted[terminal_slice]
        if len(c_term) >= 3 and np.all(c_term > 0):
            slope, intercept = np.polyfit(t_term, np.log(c_term), 1)
            terminal_points_used = len(c_term)
            pred = slope * t_term + intercept
            ss_res = float(np.sum((np.log(c_term) - pred) ** 2))
            ss_tot = float(np.sum((np.log(c_term) - np.mean(np.log(c_term))) ** 2))
            terminal_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
            if slope < 0:
                lambda_z = float(-slope)
                half_life = float(math.log(2.0) / lambda_z) if lambda_z > 0 else None
            else:
                warnings.append(
                    "Terminal slope is non-negative; lambda_z and half-life are not reliable."
                )
        else:
            warnings.append("Insufficient positive terminal points for lambda_z estimation.")
    else:
        warnings.append("Fewer than 3 positive concentration points; cannot estimate lambda_z.")

    clast = float(conc_sorted[-1])
    if lambda_z and lambda_z > 0 and clast > 0:
        auc_extrap = float(clast / lambda_z)
        auc_inf = float(auc_last + auc_extrap)
        if auc_inf > 0:
            extrapolated_fraction = float(auc_extrap / auc_inf)
            if extrapolated_fraction > 0.2:
                warnings.append(
                    "Extrapolated AUC fraction > 20%; terminal sampling may be insufficient."
                )

    route_norm = str(route or "extravascular").strip().lower()
    if route_norm not in {"iv", "extravascular"}:
        return {"summary": "route must be 'iv' or 'extravascular'.", "error": "invalid_route"}

    dose_value = None
    if dose is not None:
        try:
            dose_value = float(dose)
        except Exception:
            return {"summary": "dose must be numeric when provided.", "error": "invalid_dose"}
        if dose_value <= 0:
            return {"summary": "dose must be > 0 when provided.", "error": "invalid_dose"}

    clearance = None
    apparent_clearance = None
    volume = None
    apparent_volume = None

    if dose_value is not None and auc_inf and auc_inf > 0:
        if route_norm == "iv":
            clearance = float(dose_value / auc_inf)
            if lambda_z and lambda_z > 0:
                volume = float(clearance / lambda_z)
        else:
            apparent_clearance = float(dose_value / auc_inf)
            if lambda_z and lambda_z > 0:
                apparent_volume = float(apparent_clearance / lambda_z)

    label = subject_id.strip() if subject_id else "sample"
    hl_text = f"{half_life:.3g}" if half_life is not None else "NA"
    summary = (
        f"Basic NCA for {label}: Cmax={cmax:.4g} at Tmax={tmax:.4g}, "
        f"AUC_last={auc_last:.4g}, t1/2={hl_text}."
    )

    return {
        "summary": summary,
        "subject_id": subject_id.strip() or None,
        "n_points": int(len(times_sorted)),
        "terminal_points_used": int(terminal_points_used),
        "route": route_norm,
        "dose": _safe_round(dose_value, 6),
        "cmax": _safe_round(cmax, 6),
        "tmax": _safe_round(tmax, 6),
        "auc_last": _safe_round(auc_last, 6),
        "clast": _safe_round(clast, 6),
        "lambda_z": _safe_round(lambda_z, 6),
        "terminal_r_squared": _safe_round(terminal_r2, 6),
        "half_life": _safe_round(half_life, 6),
        "auc_extrapolated": _safe_round(auc_extrap, 6),
        "auc_inf": _safe_round(auc_inf, 6),
        "extrapolated_fraction": _safe_round(extrapolated_fraction, 6),
        "clearance": _safe_round(clearance, 6),
        "volume_distribution": _safe_round(volume, 6),
        "apparent_clearance": _safe_round(apparent_clearance, 6),
        "apparent_volume_distribution": _safe_round(apparent_volume, 6),
        "warnings": warnings,
    }
