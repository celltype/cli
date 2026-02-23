"""
Regulatory tools for CDISC delivery quality checks.

Focused on pragmatic linting of SDTM-like tabular datasets and Define-XML files.
"""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

from ct.tools import registry


_COLUMN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ISO8601_PARTIAL_RE = re.compile(
    r"^\d{4}(-\d{2}){0,2}((T\d{2}(:\d{2}){0,2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?$"
)

_DOMAIN_REQUIRED = {
    "DM": ["SUBJID", "SEX", "RFSTDTC"],
    "AE": ["AESEQ", "AETERM", "AESTDTC"],
    "LB": ["LBSEQ", "LBTEST", "LBORRES", "LBDTC"],
    "VS": ["VSSEQ", "VSTEST", "VSORRES", "VSDTC"],
    "CM": ["CMSEQ", "CMTRT", "CMSTDTC"],
    "EX": ["EXSEQ", "EXTRT", "EXSTDTC"],
    "MH": ["MHSEQ", "MHTERM"],
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _issue(
    issues: list[dict],
    severity: str,
    code: str,
    message: str,
    field: str | None = None,
) -> None:
    payload = {"severity": severity, "code": code, "message": message}
    if field:
        payload["field"] = field
    issues.append(payload)


def _score_quality(errors: int, warnings: int) -> int:
    return max(0, int(100 - errors * 12 - warnings * 4))


def _read_tabular(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv"}:
            return pd.read_csv(path), None
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t"), None
        if suffix == ".parquet":
            return pd.read_parquet(path), None
        if suffix == ".xpt":
            return pd.read_sas(path, format="xport"), None
        return None, f"Unsupported dataset format '{suffix or '<none>'}'"
    except Exception as exc:
        return None, f"Failed to read dataset: {exc}"


def _infer_domain_from_path(path: Path) -> str:
    stem = path.stem.upper()
    tokens = [t for t in re.split(r"[^A-Z0-9]+", stem) if t]
    for token in tokens:
        if token in _DOMAIN_REQUIRED:
            return token
    alpha = "".join(ch for ch in stem if ch.isalpha())
    return alpha[:2] if len(alpha) >= 2 else ""


@registry.register(
    name="regulatory.cdisc_lint",
    description="Lint a CDISC-like tabular dataset for naming, key, required-variable, and date-format issues",
    category="regulatory",
    parameters={
        "dataset_path": "Path to SDTM/ADaM-like dataset file (.csv, .tsv, .parquet, .xpt)",
        "domain": "Optional expected domain (e.g., AE, DM, LB); auto-inferred when omitted",
        "required_columns": "Optional list of additional required column names",
        "strict": "If true, treat variable-name length violations as errors",
        "max_issues": "Maximum issues returned (default 100)",
    },
    usage_guide=(
        "Use before submission-package handoff to catch high-impact CDISC data quality issues early. "
        "Best for SDTM-like domains with STUDYID/USUBJID and --DTC fields."
    ),
)
def cdisc_lint(
    dataset_path: str,
    domain: str = "",
    required_columns: list[str] | None = None,
    strict: bool = False,
    max_issues: int = 100,
    **kwargs,
) -> dict:
    """Run pragmatic CDISC-style lint checks on a tabular dataset."""
    del kwargs

    if not dataset_path:
        return {"summary": "dataset_path is required.", "error": "missing_dataset_path"}

    path = Path(dataset_path).expanduser()
    if not path.exists():
        return {"summary": f"Dataset file not found: {path}", "error": "file_not_found"}
    if path.is_dir():
        return {"summary": f"dataset_path must be a file: {path}", "error": "path_is_directory"}

    frame, read_error = _read_tabular(path)
    if read_error:
        return {"summary": read_error, "error": "read_failed"}

    assert frame is not None  # For type checkers.

    issues: list[dict] = []
    n_rows = int(len(frame))
    n_cols = int(len(frame.columns))

    requested_domain = str(domain or "").strip().upper()
    inferred_domain = requested_domain or _infer_domain_from_path(path)

    columns = [str(c).strip() for c in frame.columns.tolist()]
    dup_cols = sorted({c for c in columns if columns.count(c) > 1})
    for col in dup_cols:
        _issue(issues, "error", "duplicate_column", f"Duplicate column name: {col}", field=col)

    for col in columns:
        if not _COLUMN_RE.match(col):
            _issue(
                issues,
                "error",
                "invalid_variable_name",
                "Variable names must be uppercase A-Z, 0-9, underscore, starting with a letter.",
                field=col,
            )
        if len(col) > 8:
            sev = "error" if strict else "warning"
            _issue(
                issues,
                sev,
                "variable_name_too_long",
                f"Variable name exceeds 8 characters ({len(col)}).",
                field=col,
            )

    required = ["STUDYID", "USUBJID"]
    if inferred_domain:
        required.append("DOMAIN")
    required.extend(_DOMAIN_REQUIRED.get(inferred_domain, []))
    if required_columns:
        required.extend([str(c).strip().upper() for c in required_columns if str(c).strip()])
    required = sorted(set(required))

    missing_required = [c for c in required if c not in columns]
    for col in missing_required:
        _issue(issues, "error", "missing_required_column", "Missing required column.", field=col)

    if "DOMAIN" in columns and inferred_domain:
        observed = sorted(
            {
                str(v).strip().upper()
                for v in frame["DOMAIN"].dropna().astype(str).tolist()
                if str(v).strip()
            }
        )
        if observed and observed != [inferred_domain]:
            _issue(
                issues,
                "error",
                "domain_mismatch",
                f"DOMAIN values {observed} do not match expected domain {inferred_domain}.",
                field="DOMAIN",
            )

    if n_rows == 0:
        _issue(issues, "warning", "empty_dataset", "Dataset has zero rows.")

    # Missingness on required columns present in data.
    for col in required:
        if col not in frame.columns:
            continue
        series = frame[col]
        missing_count = int(series.isna().sum())
        if series.dtype == object:
            missing_count += int(series.astype(str).str.strip().eq("").sum())
        if missing_count > 0:
            _issue(
                issues,
                "error",
                "required_column_missing_values",
                f"{missing_count} missing/blank values in required column.",
                field=col,
            )

    key_cols = [c for c in ["STUDYID", "USUBJID"] if c in frame.columns]
    seq_col = f"{inferred_domain}SEQ" if inferred_domain else ""
    if seq_col and seq_col in frame.columns:
        key_cols.append(seq_col)
    if len(key_cols) >= 2 and n_rows > 0:
        dup_rows = int(frame.duplicated(subset=key_cols).sum())
        if dup_rows > 0:
            _issue(
                issues,
                "error",
                "duplicate_keys",
                f"{dup_rows} duplicate rows detected for key {key_cols}.",
            )

    date_cols = [c for c in columns if c.endswith("DTC")]
    for col in date_cols:
        raw = frame[col].dropna().astype(str).str.strip()
        if len(raw) == 0:
            continue
        bad = raw[~raw.str.match(_ISO8601_PARTIAL_RE, na=False)]
        if len(bad) > 0:
            _issue(
                issues,
                "error",
                "invalid_datetime_format",
                f"{len(bad)} values are not ISO8601/partial ISO8601 compliant.",
                field=col,
            )

    n_errors = sum(1 for x in issues if x["severity"] == "error")
    n_warnings = sum(1 for x in issues if x["severity"] == "warning")
    score = _score_quality(n_errors, n_warnings)
    max_issues = max(1, int(max_issues or 100))

    summary = (
        f"CDISC lint for {path.name}: {n_errors} error(s), {n_warnings} warning(s), "
        f"quality score {score}/100."
    )

    return {
        "summary": summary,
        "dataset_path": str(path),
        "domain": inferred_domain or None,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "required_columns_checked": required,
        "error_count": n_errors,
        "warning_count": n_warnings,
        "quality_score": score,
        "issues": issues[:max_issues],
    }


def _duplicate_oids(elements: list[ET.Element]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for elem in elements:
        oid = str(elem.attrib.get("OID", "")).strip()
        if not oid:
            continue
        if oid in seen:
            duplicates.add(oid)
        seen.add(oid)
    return duplicates


def _attr_by_local_name(elem: ET.Element, local_name: str) -> str:
    for key, value in elem.attrib.items():
        if _local_name(key) == local_name:
            return str(value).strip()
    return ""


@registry.register(
    name="regulatory.define_xml_lint",
    description="Lint a Define-XML file for structural integrity and common referential issues",
    category="regulatory",
    parameters={
        "define_xml_path": "Path to define.xml file",
        "strict": "If true, missing optional metadata is elevated to warning/error",
        "max_issues": "Maximum issues returned (default 100)",
    },
    usage_guide=(
        "Use to preflight Define-XML before package delivery. Checks parseability, "
        "core structure, and reference integrity (ItemRef/CodeListRef/ValueListRef)."
    ),
)
def define_xml_lint(
    define_xml_path: str,
    strict: bool = False,
    max_issues: int = 100,
    **kwargs,
) -> dict:
    """Run structural and referential lint checks for Define-XML."""
    del kwargs

    if not define_xml_path:
        return {"summary": "define_xml_path is required.", "error": "missing_define_xml_path"}

    path = Path(define_xml_path).expanduser()
    if not path.exists():
        return {"summary": f"Define-XML file not found: {path}", "error": "file_not_found"}
    if path.is_dir():
        return {"summary": f"define_xml_path must be a file: {path}", "error": "path_is_directory"}

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return {"summary": f"Invalid XML: {exc}", "error": "xml_parse_error"}
    except Exception as exc:
        return {"summary": f"Failed to read Define-XML: {exc}", "error": "read_failed"}

    root = tree.getroot()
    issues: list[dict] = []

    if _local_name(root.tag) != "ODM":
        _issue(issues, "error", "root_not_odm", "Root element is not ODM.")

    studies = [x for x in root.iter() if _local_name(x.tag) == "Study"]
    mdvs = [x for x in root.iter() if _local_name(x.tag) == "MetaDataVersion"]
    item_defs = [x for x in root.iter() if _local_name(x.tag) == "ItemDef"]
    item_groups = [x for x in root.iter() if _local_name(x.tag) == "ItemGroupDef"]
    code_lists = [x for x in root.iter() if _local_name(x.tag) == "CodeList"]
    value_lists = [x for x in root.iter() if _local_name(x.tag) == "ValueListDef"]
    where_defs = [x for x in root.iter() if _local_name(x.tag) == "WhereClauseDef"]
    leaf_nodes = [x for x in root.iter() if _local_name(x.tag).lower() == "leaf"]

    if not studies:
        _issue(issues, "error", "missing_study", "No Study element found.")
    if not mdvs:
        _issue(issues, "error", "missing_metadataversion", "No MetaDataVersion element found.")
    if not item_defs:
        _issue(issues, "error", "missing_itemdefs", "No ItemDef elements found.")
    if not item_groups:
        _issue(issues, "error", "missing_itemgroups", "No ItemGroupDef elements found.")

    for oid in sorted(_duplicate_oids(item_defs)):
        _issue(issues, "error", "duplicate_itemdef_oid", f"Duplicate ItemDef OID: {oid}")
    for oid in sorted(_duplicate_oids(item_groups)):
        _issue(issues, "error", "duplicate_itemgroup_oid", f"Duplicate ItemGroupDef OID: {oid}")
    for oid in sorted(_duplicate_oids(code_lists)):
        _issue(issues, "error", "duplicate_codelist_oid", f"Duplicate CodeList OID: {oid}")

    item_oids = {str(x.attrib.get("OID", "")).strip() for x in item_defs if x.attrib.get("OID")}
    codelist_oids = {
        str(x.attrib.get("OID", "")).strip()
        for x in code_lists
        if x.attrib.get("OID")
    }
    valuelist_oids = {
        str(x.attrib.get("OID", "")).strip()
        for x in value_lists
        if x.attrib.get("OID")
    }
    where_oids = {
        str(x.attrib.get("OID", "")).strip()
        for x in where_defs
        if x.attrib.get("OID")
    }

    for item in item_defs:
        oid = str(item.attrib.get("OID", "")).strip() or "<missing>"
        if not str(item.attrib.get("Name", "")).strip():
            _issue(issues, "error", "itemdef_missing_name", "ItemDef missing Name.", field=oid)
        if not str(item.attrib.get("DataType", "")).strip():
            _issue(issues, "error", "itemdef_missing_datatype", "ItemDef missing DataType.", field=oid)

    for ref in [x for x in root.iter() if _local_name(x.tag) == "ItemRef"]:
        item_oid = str(ref.attrib.get("ItemOID", "")).strip()
        if not item_oid:
            _issue(issues, "error", "itemref_missing_itemoid", "ItemRef missing ItemOID.")
            continue
        if item_oid not in item_oids:
            _issue(
                issues,
                "error",
                "itemref_unknown_itemoid",
                f"ItemRef points to unknown ItemDef OID: {item_oid}",
            )

    for ref in [x for x in root.iter() if _local_name(x.tag) == "CodeListRef"]:
        code_oid = str(ref.attrib.get("CodeListOID", "")).strip()
        if not code_oid:
            _issue(issues, "error", "codelistref_missing_oid", "CodeListRef missing CodeListOID.")
            continue
        if code_oid not in codelist_oids:
            _issue(
                issues,
                "error",
                "codelistref_unknown_oid",
                f"CodeListRef points to unknown CodeList OID: {code_oid}",
            )

    for ref in [x for x in root.iter() if _local_name(x.tag) == "ValueListRef"]:
        value_oid = str(ref.attrib.get("ValueListOID", "")).strip()
        if not value_oid:
            _issue(issues, "error", "valuelistref_missing_oid", "ValueListRef missing ValueListOID.")
            continue
        if value_oid not in valuelist_oids:
            _issue(
                issues,
                "error",
                "valuelistref_unknown_oid",
                f"ValueListRef points to unknown ValueListDef OID: {value_oid}",
            )

    for ref in [x for x in root.iter() if _local_name(x.tag) == "WhereClauseRef"]:
        where_oid = str(ref.attrib.get("WhereClauseOID", "")).strip()
        if not where_oid:
            _issue(issues, "warning", "whereclauseref_missing_oid", "WhereClauseRef missing OID.")
            continue
        if where_oids and where_oid not in where_oids:
            _issue(
                issues,
                "error",
                "whereclauseref_unknown_oid",
                f"WhereClauseRef points to unknown WhereClauseDef OID: {where_oid}",
            )

    if not leaf_nodes and strict:
        _issue(issues, "warning", "missing_leaf", "No def:leaf nodes found for data/metadata files.")

    for leaf in leaf_nodes:
        href = _attr_by_local_name(leaf, "href")
        if not href:
            _issue(issues, "error", "leaf_missing_href", "def:leaf is missing xlink:href.")

    n_errors = sum(1 for x in issues if x["severity"] == "error")
    n_warnings = sum(1 for x in issues if x["severity"] == "warning")
    score = _score_quality(n_errors, n_warnings)
    max_issues = max(1, int(max_issues or 100))

    summary = (
        f"Define-XML lint for {path.name}: {n_errors} error(s), {n_warnings} warning(s), "
        f"quality score {score}/100."
    )

    return {
        "summary": summary,
        "define_xml_path": str(path),
        "error_count": n_errors,
        "warning_count": n_warnings,
        "quality_score": score,
        "counts": {
            "study": len(studies),
            "metadataversion": len(mdvs),
            "itemdef": len(item_defs),
            "itemgroupdef": len(item_groups),
            "codelist": len(code_lists),
            "valuelistdef": len(value_lists),
            "leaf": len(leaf_nodes),
        },
        "issues": issues[:max_issues],
    }


def _extract_leaf_hrefs(xml_path: Path) -> tuple[list[str], str | None]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        return [], str(exc)

    hrefs = []
    for leaf in [x for x in root.iter() if _local_name(x.tag).lower() == "leaf"]:
        href = _attr_by_local_name(leaf, "href")
        if href:
            hrefs.append(href.strip())
    return hrefs, None


@registry.register(
    name="regulatory.submission_package_check",
    description="Run cross-file submission package checks across datasets and Define-XML",
    category="regulatory",
    parameters={
        "package_dir": "Submission package root directory",
        "define_xml_path": "Optional explicit define.xml path (auto-discovered when omitted)",
        "max_datasets": "Maximum dataset files to lint (default 30)",
        "strict": "If true, tighten checks (warnings promoted where applicable)",
    },
    usage_guide=(
        "Use before external handoff to validate end-to-end package consistency: "
        "Define-XML integrity, dataset linting, and cross-file reference checks."
    ),
)
def submission_package_check(
    package_dir: str,
    define_xml_path: str = "",
    max_datasets: int = 30,
    strict: bool = False,
    **kwargs,
) -> dict:
    """Run cross-file validation for a submission package directory."""
    del kwargs
    if not package_dir:
        return {"summary": "package_dir is required.", "error": "missing_package_dir"}

    root = Path(package_dir).expanduser()
    if not root.exists():
        return {"summary": f"Package directory not found: {root}", "error": "package_not_found"}
    if not root.is_dir():
        return {"summary": f"package_dir must be a directory: {root}", "error": "not_a_directory"}

    max_datasets = max(1, min(int(max_datasets or 30), 200))
    issues: list[dict] = []

    # Resolve define.xml
    define_paths = []
    if define_xml_path:
        candidate = Path(define_xml_path).expanduser()
        if not candidate.exists():
            _issue(issues, "error", "define_xml_missing", f"define_xml_path not found: {candidate}")
        else:
            define_paths = [candidate]
    else:
        define_paths = sorted(
            [p for p in root.rglob("*.xml") if "define" in p.name.lower()],
            key=lambda p: p.name.lower(),
        )
        if not define_paths:
            _issue(issues, "warning", "define_xml_not_found", "No define.xml discovered in package directory.")

    define_results = []
    for p in define_paths[:2]:
        lint = define_xml_lint(define_xml_path=str(p), strict=strict, max_issues=200)
        define_results.append(lint)
        if "error" in lint:
            _issue(issues, "error", "define_xml_lint_failed", lint["summary"], field=str(p))

    # Dataset discovery
    dataset_paths = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".xpt", ".csv", ".tsv", ".parquet"}
        ]
    )
    if not dataset_paths:
        _issue(issues, "error", "no_datasets_found", "No dataset files (*.xpt/*.csv/*.tsv/*.parquet) found.")

    dataset_lints = []
    for p in dataset_paths[:max_datasets]:
        result = cdisc_lint(
            dataset_path=str(p),
            strict=strict,
            max_issues=50,
        )
        dataset_lints.append(result)
        if result.get("error_count", 0) > 0:
            _issue(
                issues,
                "error",
                "dataset_errors",
                f"{p.name}: {result.get('error_count', 0)} error(s) from CDISC lint.",
                field=str(p),
            )

    # Cross-file leaf href resolution against package files.
    referenced_files = set()
    missing_leaf_files = []
    for d in define_results:
        define_path = d.get("define_xml_path")
        if not define_path:
            continue
        hrefs, href_error = _extract_leaf_hrefs(Path(define_path))
        if href_error:
            _issue(
                issues,
                "warning",
                "leaf_parse_failed",
                f"Could not parse leaf href values for {define_path}: {href_error}",
            )
            continue
        base_dir = Path(define_path).parent
        for href in hrefs:
            normalized = href.replace("\\", "/").strip()
            referenced_files.add(normalized)
            candidate = (base_dir / normalized).resolve()
            if not candidate.exists():
                # Fallback check relative to package root.
                candidate = (root / normalized).resolve()
            if not candidate.exists():
                missing_leaf_files.append(normalized)

    for href in sorted(set(missing_leaf_files)):
        _issue(
            issues,
            "error",
            "missing_leaf_target",
            f"Define-XML leaf href does not resolve to a file in package: {href}",
        )

    # Orphan dataset warning: files that are present but not referenced in define.xml.
    if referenced_files:
        dataset_relpaths = {
            str(p.relative_to(root)).replace("\\", "/"): p
            for p in dataset_paths
            if p.is_file()
        }
        unreferenced = []
        ref_suffixes = {Path(x).name.lower() for x in referenced_files}
        for rel, path_obj in dataset_relpaths.items():
            if path_obj.name.lower() not in ref_suffixes:
                unreferenced.append(rel)
        for rel in sorted(unreferenced)[:20]:
            _issue(
                issues,
                "warning",
                "unreferenced_dataset",
                f"Dataset not referenced by define.xml leaf entries: {rel}",
            )

    n_errors = sum(1 for x in issues if x["severity"] == "error")
    n_warnings = sum(1 for x in issues if x["severity"] == "warning")
    readiness_score = _score_quality(n_errors, n_warnings)
    if n_errors == 0 and readiness_score >= 85:
        readiness = "ready"
    elif n_errors <= 3 and readiness_score >= 60:
        readiness = "needs_review"
    else:
        readiness = "not_ready"

    summary = (
        f"Submission package check for {root}: {readiness} "
        f"(score={readiness_score}/100, errors={n_errors}, warnings={n_warnings}). "
        f"Datasets linted: {len(dataset_lints)}/{len(dataset_paths)}."
    )

    return {
        "summary": summary,
        "package_dir": str(root),
        "readiness": readiness,
        "readiness_score": readiness_score,
        "error_count": n_errors,
        "warning_count": n_warnings,
        "define_xml_results": define_results,
        "dataset_lints": dataset_lints,
        "datasets_discovered": len(dataset_paths),
        "datasets_linted": len(dataset_lints),
        "issues": issues,
    }
