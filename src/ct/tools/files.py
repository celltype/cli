"""
File I/O tools for ct.

Read files, write reports/CSV, edit/create/delete files, and search file contents.
Restricted to safe directories (~/.ct/, config-specified paths, and CWD).
"""

import csv
import io
import re
import shutil
from pathlib import Path

from ct.tools import registry


def _allowed_paths(config=None) -> list[Path]:
    """Return list of directories the user is allowed to read from."""
    allowed = [Path.home() / ".ct"]

    if config:
        for key in ("data.base", "data.depmap", "data.prism", "data.l1000",
                     "data.msigdb", "sandbox.output_dir"):
            val = config.get(key)
            if val:
                p = Path(val)
                # Add the directory (or parent if it's a file path)
                allowed.append(p if p.is_dir() else p.parent)

        # Also allow sandbox extra read dirs (e.g. capsule data directories)
        extra = config.get("sandbox.extra_read_dirs")
        if extra:
            for d in str(extra).split(","):
                d = d.strip()
                if d:
                    allowed.append(Path(d))

    return allowed


def _is_allowed(path: Path, config=None) -> bool:
    """Check if a path is within allowed directories."""
    resolved = path.resolve()
    for allowed in _allowed_paths(config):
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue
    return False


def _is_within_cwd(path: Path) -> bool:
    """Check if a resolved path is under the current working directory.

    Resolves symlinks before checking to prevent traversal via symlinks
    (e.g., ./data -> /etc would be rejected).
    """
    try:
        resolved = path.resolve(strict=False)
        cwd = Path.cwd().resolve()
        resolved.relative_to(cwd)
        # Extra check: if path contains a symlink, verify each component
        # stays within CWD after resolution
        if path.is_symlink():
            target = path.resolve()
            target.relative_to(cwd)
        return True
    except ValueError:
        return False


# Paths that must never be edited or deleted
_PROTECTED_PATTERNS = (
    "/.git/",
    ".env",
)


def _is_protected(path: Path) -> bool:
    """Check if a path is in the protected blocklist."""
    resolved = str(path.resolve())
    name = path.name.lower()

    # .ssh — block private keys but allow .pub
    if ".ssh/" in resolved or ".ssh\\" in resolved:
        # Allow public keys
        if name.endswith(".pub"):
            return False
        return True

    for pattern in _PROTECTED_PATTERNS:
        if pattern in resolved:
            return True

    # Block private keys outside .ssh too
    if name.startswith("id_") and not name.endswith(".pub"):
        return True
    return False


def _output_dir(config=None) -> Path:
    """Get the output directory, creating it if needed."""
    if config:
        out = config.get("sandbox.output_dir")
        if out:
            p = Path(out)
            p.mkdir(parents=True, exist_ok=True)
            return p
    default = Path.cwd() / "outputs"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _resolve_output_path(out_dir: Path, filename: str) -> tuple[Path | None, str | None]:
    """Resolve an output filename safely within out_dir."""
    raw_name = (filename or "").strip()
    if not raw_name:
        return None, "Filename cannot be empty."

    rel_path = Path(raw_name)
    if rel_path.is_absolute():
        return None, "Absolute paths are not allowed."

    resolved = (out_dir / rel_path).resolve()
    try:
        resolved.relative_to(out_dir.resolve())
    except ValueError:
        return None, "Path traversal detected."

    if resolved.name in {"", ".", ".."}:
        return None, "Filename must point to a file."

    return resolved, None


def _resolve_cwd_path(path: str) -> tuple[Path | None, str | None]:
    """Resolve path and enforce current-working-directory containment."""
    p = Path(path).expanduser()
    if not _is_within_cwd(p):
        return None, "path_not_allowed"
    return p, None


@registry.register(
    name="files.read_file",
    description="Read a text file and return its contents",
    category="files",
    parameters={"path": "Path to the file to read"},
    usage_guide=(
        "Use to read data files, prior reports, configuration files, or any file in the "
        "current working directory. Also reads from ~/.ct/ and configured data directories."
    ),
)
def read_file(path: str, _session=None, **kwargs) -> dict:
    """Read a text file and return its contents."""
    config = _session.config if _session else None
    p = Path(path).expanduser()

    if not _is_allowed(p, config) and not _is_within_cwd(p):
        return {
            "summary": f"Access denied: {path} is outside allowed directories.",
            "error": "path_not_allowed",
        }

    if _is_protected(p):
        return {
            "summary": f"Access denied: {path} is a protected file.",
            "error": "path_protected",
        }

    if not p.exists():
        return {
            "summary": f"File not found: {path}",
            "error": "file_not_found",
        }

    try:
        content = p.read_text(encoding="utf-8")
        lines = content.count("\n") + 1
        return {
            "summary": f"Read {p.name} ({lines} lines, {len(content)} chars).",
            "path": str(p),
            "content": content,
            "lines": lines,
        }
    except UnicodeDecodeError:
        # Handle common binary tabular formats with a structured preview
        # rather than hard-failing decode.
        suffix = p.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            try:
                import pandas as pd

                xls = pd.ExcelFile(p)
                sheet_names = xls.sheet_names
                if not sheet_names:
                    return {
                        "summary": f"Excel file has no sheets: {p.name}",
                        "path": str(p),
                        "sheets": [],
                    }

                df = pd.read_excel(p, sheet_name=sheet_names[0], nrows=50)
                rows = len(df)
                cols = [str(c) for c in df.columns]
                preview = df.head(5).to_dict(orient="records")
                return {
                    "summary": (
                        f"Read Excel file {p.name}: {len(sheet_names)} sheet(s), "
                        f"previewed '{sheet_names[0]}' ({rows} rows, {len(cols)} columns in preview)."
                    ),
                    "path": str(p),
                    "format": "excel",
                    "sheets": sheet_names,
                    "sheet": sheet_names[0],
                    "columns": cols,
                    "rows_previewed": rows,
                    "preview": preview,
                }
            except Exception as e:
                return {"summary": f"Error reading Excel file {path}: {e}", "error": str(e)}
        return {
            "summary": (
                f"{p.name} appears to be a binary/non-UTF8 file. "
                "Use code.execute or a format-specific tool to parse it."
            ),
            "path": str(p),
            "error": "binary_file",
        }
    except Exception as e:
        return {"summary": f"Error reading {path}: {e}", "error": str(e)}


@registry.register(
    name="files.edit_file",
    description="Edit a file by replacing an exact string match with new content",
    category="files",
    parameters={
        "path": "Path to the file to edit (must be within CWD)",
        "old_string": "Exact string to find and replace (must be unique in the file)",
        "new_string": "Replacement string",
    },
    usage_guide=(
        "Use to make targeted edits to files in the current working directory. "
        "The old_string must appear exactly once in the file for unambiguous replacement."
    ),
)
def edit_file(path: str, old_string: str, new_string: str, **kwargs) -> dict:
    """Edit a file by exact string replacement."""
    p = Path(path).expanduser()

    if not _is_within_cwd(p):
        return {"summary": f"Access denied: {path} is outside working directory.", "error": "path_not_allowed"}
    if _is_protected(p):
        return {"summary": f"Protected path: {path} cannot be edited.", "error": "path_protected"}
    if not p.exists():
        return {"summary": f"File not found: {path}", "error": "file_not_found"}

    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return {"summary": f"Error reading {path}: {e}", "error": str(e)}

    count = content.count(old_string)
    if count == 0:
        return {"summary": f"String not found in {p.name}.", "error": "string_not_found"}
    if count > 1:
        return {
            "summary": f"Ambiguous: '{old_string[:50]}...' appears {count} times in {p.name}. Provide more context.",
            "error": "ambiguous_match",
            "match_count": count,
        }

    new_content = content.replace(old_string, new_string, 1)
    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return {"summary": f"Error writing {path}: {e}", "error": str(e)}

    return {
        "summary": f"Edited {p.name}: replaced {len(old_string)} chars with {len(new_string)} chars.",
        "path": str(p.resolve()),
        "old_length": len(old_string),
        "new_length": len(new_string),
    }


@registry.register(
    name="files.create_file",
    description="Create a new file with the given content",
    category="files",
    parameters={
        "path": "Path for the new file (must be within CWD)",
        "content": "Content to write to the file",
    },
    usage_guide=(
        "Use to create new files (scripts, configs, data files) in the working directory. "
        "Will not overwrite existing files — use edit_file for modifications."
    ),
)
def create_file(path: str, content: str, **kwargs) -> dict:
    """Create a new file. Refuses to overwrite existing files."""
    p = Path(path).expanduser()

    if not _is_within_cwd(p):
        return {"summary": f"Access denied: {path} is outside working directory.", "error": "path_not_allowed"}
    if _is_protected(p):
        return {"summary": f"Protected path: {path} cannot be created.", "error": "path_protected"}
    if p.exists():
        try:
            existing = p.read_text(encoding="utf-8")
            if existing == content:
                lines = content.count("\n") + 1
                return {
                    "summary": f"File already exists with identical content: {p.name}.",
                    "path": str(p.resolve()),
                    "lines": lines,
                    "size": len(content),
                    "unchanged": True,
                }
            # Auto-update stale generated artifacts so repeated workflows are idempotent.
            p.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            return {
                "summary": f"Updated existing file {p.name} ({lines} lines, {len(content)} chars).",
                "path": str(p.resolve()),
                "lines": lines,
                "size": len(content),
                "overwritten": True,
            }
        except Exception:
            # Keep default behavior for non-text/unreadable files.
            pass
        return {"summary": f"File already exists: {path}. Use edit_file to modify.", "error": "file_exists"}

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        return {"summary": f"Error creating {path}: {e}", "error": str(e)}

    lines = content.count("\n") + 1
    return {
        "summary": f"Created {p.name} ({lines} lines, {len(content)} chars).",
        "path": str(p.resolve()),
        "lines": lines,
        "size": len(content),
    }


@registry.register(
    name="files.delete_file",
    description="Delete a file from the working directory",
    category="files",
    parameters={"path": "Path to the file to delete (must be within CWD)"},
    usage_guide="Use to remove files from the working directory. Cannot delete directories.",
)
def delete_file(path: str, **kwargs) -> dict:
    """Delete a single file."""
    p = Path(path).expanduser()

    if not _is_within_cwd(p):
        return {"summary": f"Access denied: {path} is outside working directory.", "error": "path_not_allowed"}
    if _is_protected(p):
        return {"summary": f"Protected path: {path} cannot be deleted.", "error": "path_protected"}
    if not p.exists():
        return {"summary": f"File not found: {path}", "error": "file_not_found"}
    if p.is_dir():
        return {"summary": f"Cannot delete directory: {path}. Only files.", "error": "is_directory"}

    try:
        size = p.stat().st_size
        p.unlink()
    except Exception as e:
        return {"summary": f"Error deleting {path}: {e}", "error": str(e)}

    return {
        "summary": f"Deleted {p.name} ({size} bytes).",
        "path": str(p.resolve()),
    }


@registry.register(
    name="files.move_file",
    description="Move or rename a file within the working directory",
    category="files",
    parameters={
        "source_path": "Path to source file (must be within CWD)",
        "dest_path": "Path to destination file (must be within CWD)",
        "overwrite": "Whether to overwrite destination if it exists (default false)",
    },
    usage_guide=(
        "Use to rename files or reorganize outputs in the workspace. "
        "Both source and destination must stay inside the current working directory."
    ),
)
def move_file(source_path: str, dest_path: str, overwrite: bool = False, **kwargs) -> dict:
    """Move a file safely within CWD."""
    src, err = _resolve_cwd_path(source_path)
    if err:
        return {"summary": f"Access denied: {source_path} is outside working directory.", "error": err}
    dst, err = _resolve_cwd_path(dest_path)
    if err:
        return {"summary": f"Access denied: {dest_path} is outside working directory.", "error": err}
    if _is_protected(src) or _is_protected(dst):
        return {"summary": "Protected path cannot be moved.", "error": "path_protected"}
    if not src.exists():
        return {"summary": f"File not found: {source_path}", "error": "file_not_found"}
    if src.is_dir():
        return {"summary": f"Source is a directory: {source_path}", "error": "is_directory"}
    if dst.exists() and not overwrite:
        return {"summary": f"Destination exists: {dest_path}", "error": "file_exists"}
    if dst.exists() and dst.is_dir():
        return {"summary": f"Destination is a directory: {dest_path}", "error": "is_directory"}
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
    except Exception as e:
        return {"summary": f"Error moving file: {e}", "error": str(e)}
    return {"summary": f"Moved {src.name} to {dst}", "source": str(src), "destination": str(dst)}


@registry.register(
    name="files.copy_file",
    description="Copy a file within the working directory",
    category="files",
    parameters={
        "source_path": "Path to source file (must be within CWD)",
        "dest_path": "Path to destination file (must be within CWD)",
        "overwrite": "Whether to overwrite destination if it exists (default false)",
    },
    usage_guide=(
        "Use to duplicate templates, data files, or reports in the workspace "
        "without editing the original."
    ),
)
def copy_file(source_path: str, dest_path: str, overwrite: bool = False, **kwargs) -> dict:
    """Copy a file safely within CWD."""
    src, err = _resolve_cwd_path(source_path)
    if err:
        return {"summary": f"Access denied: {source_path} is outside working directory.", "error": err}
    dst, err = _resolve_cwd_path(dest_path)
    if err:
        return {"summary": f"Access denied: {dest_path} is outside working directory.", "error": err}
    if _is_protected(src) or _is_protected(dst):
        return {"summary": "Protected path cannot be copied.", "error": "path_protected"}
    if not src.exists():
        return {"summary": f"File not found: {source_path}", "error": "file_not_found"}
    if src.is_dir():
        return {"summary": f"Source is a directory: {source_path}", "error": "is_directory"}
    if dst.exists() and not overwrite:
        return {"summary": f"Destination exists: {dest_path}", "error": "file_exists"}
    if dst.exists() and dst.is_dir():
        return {"summary": f"Destination is a directory: {dest_path}", "error": "is_directory"}
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        return {"summary": f"Error copying file: {e}", "error": str(e)}
    return {"summary": f"Copied {src.name} to {dst}", "source": str(src), "destination": str(dst)}


@registry.register(
    name="files.create_directory",
    description="Create a directory within the working directory",
    category="files",
    parameters={
        "path": "Directory path to create (must be within CWD)",
        "exist_ok": "If true, do not error when directory already exists (default true)",
    },
    usage_guide="Use to create folders for outputs, reports, and structured project organization.",
)
def create_directory(path: str, exist_ok: bool = True, **kwargs) -> dict:
    """Create a directory safely within CWD."""
    p, err = _resolve_cwd_path(path)
    if err:
        return {"summary": f"Access denied: {path} is outside working directory.", "error": err}
    if _is_protected(p):
        return {"summary": f"Protected path: {path} cannot be created.", "error": "path_protected"}
    if p.exists() and p.is_file():
        return {"summary": f"Path exists as a file: {path}", "error": "is_file"}
    try:
        p.mkdir(parents=True, exist_ok=bool(exist_ok))
    except FileExistsError:
        return {"summary": f"Directory already exists: {path}", "error": "file_exists"}
    except Exception as e:
        return {"summary": f"Error creating directory: {e}", "error": str(e)}
    return {"summary": f"Directory ready: {p}", "path": str(p.resolve())}


@registry.register(
    name="files.extract_archive",
    description=(
        "Extract a ZIP, tar, tar.gz, or tar.bz2 archive. "
        "Supports extracting the full archive or specific files by pattern."
    ),
    category="files",
    parameters={
        "path": "Path to the archive file",
        "destination": "Directory to extract into (default: current working directory)",
        "pattern": "Optional glob pattern to extract only matching files (e.g. '*.mafft', '156083at2759*')",
    },
    usage_guide=(
        "Use to extract ZIP, tar, tar.gz, or tar.bz2 archives. Safer and more reliable "
        "than shell.run for archive extraction. Supports selective extraction via pattern."
    ),
)
def extract_archive(
    path: str,
    destination: str = ".",
    pattern: str = "",
    _session=None,
    **kwargs,
) -> dict:
    """Extract an archive file."""
    import fnmatch
    import tarfile
    import zipfile
    import logging
    _log = logging.getLogger("ct.tools.files")
    _log.debug("extract_archive: path=%r destination=%r pattern=%r kwargs=%r", path, destination, pattern, kwargs)

    src = Path(path).expanduser()
    if not src.exists():
        # Try relative to extra_read_dirs
        config = _session.config if _session else None
        if config:
            extra = config.get("sandbox.extra_read_dirs")
            if extra:
                for d in str(extra).split(","):
                    candidate = Path(d.strip()) / path
                    if candidate.exists():
                        src = candidate
                        break
    if not src.exists():
        return {"summary": f"Archive not found: {path}", "error": "file_not_found"}

    # Sanitize destination: only allow relative paths under CWD
    dest = Path(destination)
    cwd = Path.cwd()
    if dest.is_absolute():
        try:
            dest.resolve().relative_to(cwd.resolve())
        except ValueError:
            # Absolute path outside CWD — ignore it, use CWD
            _log.warning("Ignoring absolute destination %s, extracting to CWD", dest)
            dest = cwd
    else:
        # Relative path — resolve relative to CWD
        dest = cwd / dest
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        _log.warning("Permission denied for %s, falling back to CWD", dest)
        dest = cwd

    extracted = []
    try:
        if src.suffix == ".zip":
            with zipfile.ZipFile(src, "r") as zf:
                members = zf.namelist()
                if pattern:
                    members = [m for m in members if fnmatch.fnmatch(Path(m).name, pattern)]
                for m in members:
                    zf.extract(m, dest)
                    extracted.append(m)
        elif src.suffix in (".gz", ".bz2", ".xz", ".tar") or ".tar." in src.name:
            with tarfile.open(src, "r:*") as tf:
                members = tf.getnames()
                if pattern:
                    members = [m for m in members if fnmatch.fnmatch(Path(m).name, pattern)]
                    for m in members:
                        tf.extract(m, dest, filter="data")
                        extracted.append(m)
                else:
                    tf.extractall(dest, filter="data")
                    extracted = members
        else:
            return {"summary": f"Unsupported archive format: {src.suffix}", "error": "unsupported_format"}
    except Exception as e:
        return {"summary": f"Extraction error: {e}", "error": str(e)}

    summary = f"Extracted {len(extracted)} files from {src.name} to {dest}"
    if pattern:
        summary += f" (pattern: {pattern})"

    return {
        "summary": summary,
        "extracted_count": len(extracted),
        "destination": str(dest.resolve()),
        "files": extracted[:50],  # Cap for large archives
    }


@registry.register(
    name="files.list_directory",
    description="List directory entries with metadata",
    category="files",
    parameters={
        "path": "Directory path to inspect (default CWD)",
        "recursive": "If true, recurse through subdirectories (default false)",
        "max_entries": "Maximum entries to return (default 200)",
        "show_hidden": "Include dotfiles/directories (default false)",
    },
    usage_guide="Use to inspect workspace structure before reading or modifying files.",
)
def list_directory(
    path: str = "",
    recursive: bool = False,
    max_entries: int = 200,
    show_hidden: bool = False,
    **kwargs,
) -> dict:
    """List directory contents in a safe, bounded way."""
    base = Path(path).expanduser() if path else Path.cwd()
    # If absolute path is outside CWD or doesn't exist, try the basename under CWD
    if not base.exists() or not _is_within_cwd(base):
        if path:
            cwd_candidate = Path.cwd() / Path(path).name
            if cwd_candidate.exists() and cwd_candidate.is_dir():
                base = cwd_candidate
    if not _is_within_cwd(base) and base != Path.cwd():
        # Check if it's in allowed paths (extra_read_dirs, data dirs)
        config = getattr(kwargs.get("_session"), "config", None) if kwargs.get("_session") else None
        if not _is_allowed(base, config):
            return {"summary": f"Access denied: {path} is outside working directory.", "error": "path_not_allowed"}
    if not base.exists():
        return {"summary": f"Path not found: {base}", "error": "file_not_found"}
    if not base.is_dir():
        return {"summary": f"Not a directory: {base}", "error": "not_directory"}

    max_entries = min(max(int(max_entries), 1), 1000)
    cwd = Path.cwd().resolve()
    entries = []

    iterator = base.rglob("*") if recursive else base.iterdir()
    try:
        for p in sorted(iterator):
            name = p.name
            if not show_hidden and name.startswith("."):
                continue
            try:
                rel = str(p.resolve().relative_to(cwd))
            except ValueError:
                continue
            item = {
                "path": rel,
                "name": name,
                "type": "dir" if p.is_dir() else "file",
            }
            if p.is_file():
                try:
                    item["size"] = p.stat().st_size
                except OSError:
                    item["size"] = None
            entries.append(item)
            if len(entries) >= max_entries:
                break
    except Exception as e:
        return {"summary": f"Error listing directory: {e}", "error": str(e)}

    return {
        "summary": f"Listed {len(entries)} entries under {base}",
        "entries": entries,
        "count": len(entries),
        "directory": str(base.resolve()),
    }


@registry.register(
    name="files.search_files",
    description="Search for files by glob pattern within the working directory",
    category="files",
    parameters={
        "pattern": "Glob pattern (e.g., '**/*.py', '*.csv', 'src/**/*.ts')",
        "path": "Subdirectory to search in (default: CWD)",
    },
    usage_guide="Use to find files by name pattern. Returns file paths, names, and sizes.",
)
def search_files(pattern: str, path: str = "", **kwargs) -> dict:
    """Glob-based file search within CWD."""
    base = Path(path).expanduser() if path else Path.cwd()

    if not _is_within_cwd(base) and base != Path.cwd():
        return {"summary": f"Access denied: {path} is outside working directory.", "error": "path_not_allowed"}

    try:
        cwd = Path.cwd().resolve()
        matches = []
        for p in sorted(base.glob(pattern)):
            if p.is_file():
                try:
                    rel = str(p.resolve().relative_to(cwd))
                except ValueError:
                    continue  # Skip files outside CWD
                matches.append({
                    "path": str(p.resolve()),
                    "name": p.name,
                    "relative": rel,
                    "size": p.stat().st_size,
                })
                if len(matches) >= 100:
                    break
    except Exception as e:
        return {"summary": f"Search error: {e}", "error": str(e)}

    if not matches:
        return {"summary": f"No files matching '{pattern}'.", "files": [], "count": 0}

    listing = "\n".join(f"  {m['relative']} ({m['size']} bytes)" for m in matches[:20])
    more = f"\n  ... and {len(matches) - 20} more" if len(matches) > 20 else ""
    return {
        "summary": f"Found {len(matches)} files matching '{pattern}':\n{listing}{more}",
        "files": matches,
        "count": len(matches),
    }


@registry.register(
    name="files.search_content",
    description="Search file contents by regex pattern (like grep)",
    category="files",
    parameters={
        "pattern": "Regex pattern to search for",
        "path": "Subdirectory to search in (default: CWD)",
        "glob": "File glob filter (default: '**/*')",
        "max_results": "Maximum matches to return (default: 50)",
    },
    usage_guide=(
        "Use to search for text patterns across files — find function definitions, "
        "variable usage, TODOs, error messages, etc. Skips binary and large files."
    ),
)
def search_content(pattern: str, path: str = "", glob: str = "**/*",
                   max_results: int = 50, **kwargs) -> dict:
    """Regex content search across files in CWD."""
    base = Path(path).expanduser() if path else Path.cwd()

    if not _is_within_cwd(base) and base != Path.cwd():
        return {"summary": f"Access denied: {path} is outside working directory.", "error": "path_not_allowed"}

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return {"summary": f"Invalid regex: {e}", "error": str(e)}

    cwd = Path.cwd().resolve()
    matches = []
    files_searched = 0

    try:
        for fp in sorted(base.glob(glob)):
            if not fp.is_file():
                continue
            # Skip large files (>1MB) and likely binary files
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if size > 1_000_000:
                continue
            # Skip common binary extensions
            if fp.suffix.lower() in ('.pyc', '.pyo', '.so', '.dll', '.exe', '.bin',
                                     '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
                                     '.zip', '.tar', '.gz', '.bz2', '.whl', '.egg'):
                continue

            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            files_searched += 1
            for line_num, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    try:
                        rel = str(fp.resolve().relative_to(cwd))
                    except ValueError:
                        continue
                    preview = line.strip()
                    if len(preview) > 200:
                        preview = preview[:197] + "..."
                    matches.append({
                        "file": rel,
                        "line": line_num,
                        "text": preview,
                    })
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break
    except Exception as e:
        return {"summary": f"Search error: {e}", "error": str(e)}

    if not matches:
        return {
            "summary": f"No matches for '{pattern}' in {files_searched} files.",
            "matches": [],
            "count": 0,
            "files_searched": files_searched,
        }

    listing = "\n".join(f"  {m['file']}:{m['line']}: {m['text']}" for m in matches[:15])
    more = f"\n  ... and {len(matches) - 15} more" if len(matches) > 15 else ""
    return {
        "summary": f"Found {len(matches)} matches for '{pattern}' across {files_searched} files:\n{listing}{more}",
        "matches": matches,
        "count": len(matches),
        "files_searched": files_searched,
    }


@registry.register(
    name="files.write_report",
    description="Write a report to the output directory",
    category="files",
    parameters={
        "content": "Report content (markdown text)",
        "filename": "Output filename (e.g., 'report.md')",
        "format": "Output format: 'markdown' (default) or 'text'",
        "overwrite": "Whether to overwrite existing file (default False)",
    },
    usage_guide=(
        "Use to save analysis results as a formatted report. "
        "Output goes to the configured output directory (./outputs by default)."
    ),
)
def write_report(content: str, filename: str = "report.md",
                 format: str = "markdown", overwrite: bool = False,
                 _session=None, **kwargs) -> dict:
    """Write a report to the output directory."""
    config = _session.config if _session else None
    out_dir = _output_dir(config)

    # Ensure filename has appropriate extension
    if format == "markdown" and not filename.endswith((".md", ".markdown")):
        filename = filename + ".md"

    out_path, error = _resolve_output_path(out_dir, filename)
    if error:
        return {
            "summary": f"Invalid filename '{filename}': {error}",
            "error": "invalid_filename",
        }

    if not overwrite and out_path.exists():
        suffix = "".join(out_path.suffixes)
        stem = out_path.name[: -len(suffix)] if suffix else out_path.name
        counter = 2
        candidate = out_path.parent / f"{stem}_{counter}{suffix}"
        while candidate.exists():
            counter += 1
            candidate = out_path.parent / f"{stem}_{counter}{suffix}"
        out_path = candidate

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return {
            "summary": f"Report saved to {out_path}",
            "path": str(out_path),
            "size": len(content),
        }
    except Exception as e:
        return {"summary": f"Error writing report: {e}", "error": str(e)}


@registry.register(
    name="files.write_csv",
    description="Write structured data as a CSV file",
    category="files",
    parameters={
        "data": "List of dicts to write (each dict = one row)",
        "filename": "Output filename (e.g., 'results.csv')",
    },
    usage_guide=(
        "Use to export structured results (tables, rankings, gene lists) as CSV. "
        "Input is a list of dicts; keys become column headers."
    ),
)
def write_csv(data: list, filename: str = "results.csv",
              _session=None, **kwargs) -> dict:
    """Write structured data as CSV."""
    config = _session.config if _session else None
    out_dir = _output_dir(config)

    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    out_path, error = _resolve_output_path(out_dir, filename)
    if error:
        return {
            "summary": f"Invalid filename '{filename}': {error}",
            "error": "invalid_filename",
        }

    if not data:
        return {"summary": "No data to write.", "error": "empty_data"}

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Get column headers from first row
        if isinstance(data[0], dict):
            fieldnames = list(data[0].keys())
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            out_path.write_text(buf.getvalue(), encoding="utf-8")
        else:
            # Fallback: list of lists
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerows(data)
            out_path.write_text(buf.getvalue(), encoding="utf-8")

        return {
            "summary": f"CSV saved to {out_path} ({len(data)} rows).",
            "path": str(out_path),
            "rows": len(data),
        }
    except Exception as e:
        return {"summary": f"Error writing CSV: {e}", "error": str(e)}


@registry.register(
    name="files.list_outputs",
    description="List all files in the output directory",
    category="files",
    parameters={},
    usage_guide="Use to see what reports and exports have been generated this session.",
)
def list_outputs(_session=None, **kwargs) -> dict:
    """List all files in the output directory."""
    config = _session.config if _session else None
    out_dir = _output_dir(config)

    files = []
    if out_dir.exists():
        for p in sorted(out_dir.iterdir()):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "size": p.stat().st_size,
                    "path": str(p),
                })

    if not files:
        return {"summary": f"Output directory is empty: {out_dir}", "files": []}

    listing = "\n".join(f"  {f['name']} ({f['size']} bytes)" for f in files)
    return {
        "summary": f"Output directory ({out_dir}):\n{listing}",
        "files": files,
        "directory": str(out_dir),
    }
