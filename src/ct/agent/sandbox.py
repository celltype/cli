"""
Sandboxed Python execution environment for ct code generation.

Provides a safe exec() environment with access to scientific Python libraries
and loaded datasets. Used by the code.execute tool.
"""

import io
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


# Suppress matplotlib warnings before import
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Force non-interactive matplotlib backend before any import
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass  # matplotlib is a required dep; deferred error in _setup_namespace()


# Modules blocked from import inside the sandbox
_BLOCKED_MODULES = frozenset({
    "subprocess", "shutil", "socket", "http.server", "smtplib",
    "ctypes",
})

# Allow os import but we'll provide a safe subset in namespace
_SAFE_OS_ATTRS = frozenset({
    "path", "listdir", "walk", "getcwd", "sep", "linesep",
    "stat", "fstat", "scandir", "DirEntry",
})


def _make_safe_import(real_import):
    """Wrap __import__ to block dangerous modules."""
    def _safe_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if name in _BLOCKED_MODULES or base in _BLOCKED_MODULES:
            raise ImportError(
                f"Import of '{name}' is blocked in the ct sandbox for safety. "
                f"Use pre-built tools for operations requiring system access."
            )
        return real_import(name, *args, **kwargs)
    return _safe_import


def _is_within(path: Path, root: Path) -> bool:
    """Return True if path is located under root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _make_safe_open(output_dir: Path, extra_read_dirs: list[Path] = None):
    """Restrict sandbox file I/O to safe paths."""
    real_open = open
    output_root = output_dir.resolve()
    cwd_root = Path.cwd().resolve()
    extra_roots = [d.resolve() for d in (extra_read_dirs or [])]

    def _safe_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            # Allow file descriptors (stdout/stderr) used by internal libs.
            return real_open(file, mode, *args, **kwargs)

        path = Path(file).expanduser()
        resolved = path.resolve() if path.is_absolute() else (cwd_root / path).resolve()

        downloads_root = (Path.home() / ".ct" / "downloads").resolve()
        tmp_root = Path("/tmp").resolve()
        can_read = (
            _is_within(resolved, cwd_root)
            or _is_within(resolved, output_root)
            or _is_within(resolved, downloads_root)
            or _is_within(resolved, tmp_root)
            or any(_is_within(resolved, d) for d in extra_roots)
        )
        if not can_read:
            raise PermissionError(
                f"Sandbox file reads are restricted to {cwd_root}, {output_root}, and {downloads_root}"
            )

        writes = any(flag in mode for flag in ("w", "a", "x", "+"))
        tmp_root = Path("/tmp").resolve()
        if writes and not (
            _is_within(resolved, output_root)
            or _is_within(resolved, tmp_root)
        ):
            raise PermissionError(
                f"Sandbox file writes are restricted to {output_root} and /tmp"
            )

        if writes:
            resolved.parent.mkdir(parents=True, exist_ok=True)

        return real_open(resolved, mode, *args, **kwargs)

    return _safe_open


_ALLOWED_SUBPROCESS_COMMANDS = frozenset({
    "bwa", "samtools", "busco", "minimap2", "bowtie2",
    "muscle", "mafft", "clustalw", "phykit",
})


def _make_safe_subprocess(allowed_commands: frozenset[str] = _ALLOWED_SUBPROCESS_COMMANDS):
    """Create a restricted subprocess.run that only allows whitelisted bioinformatics commands."""
    import subprocess as _subprocess

    def safe_subprocess_run(cmd, **kwargs):
        """Run a whitelisted command. Only bwa, samtools, busco, minimap2, bowtie2 are allowed."""
        if not cmd:
            raise PermissionError("Empty command")
        binary = str(cmd[0]).split("/")[-1]  # handle full paths
        if binary not in allowed_commands:
            raise PermissionError(
                f"Command '{binary}' not in allowed list: {sorted(allowed_commands)}. "
                f"Only bioinformatics CLI tools are permitted."
            )
        kwargs.setdefault("timeout", 600)
        kwargs.setdefault("capture_output", True)
        # Use text mode with error handling for non-UTF8 output
        if "text" not in kwargs and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
            kwargs["errors"] = "replace"
        return _subprocess.run(cmd, **kwargs)

    return safe_subprocess_run


class Sandbox:
    """Sandboxed execution environment for generated Python code."""

    def __init__(self, timeout: int = 30, output_dir: Path = None, max_retries: int = 2,
                 extra_read_dirs: list[Path] = None):
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "outputs"
        self.max_retries = max_retries
        self.extra_read_dirs = [Path(d).resolve() for d in (extra_read_dirs or [])]
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self):
        """Populate namespace with safe libraries and utilities."""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import json
        import re
        import math
        import collections
        import itertools
        import datetime
        import zipfile
        import glob as glob_mod
        import io
        import tempfile
        import struct
        import csv
        import gzip
        import os
        import os.path
        import urllib.request

        # Build safe builtins dict
        if isinstance(__builtins__, dict):
            safe_builtins = dict(__builtins__)
        else:
            safe_builtins = {k: getattr(__builtins__, k) for k in dir(__builtins__)}
        safe_builtins["__import__"] = _make_safe_import(__import__)
        safe_builtins["open"] = _make_safe_open(self.output_dir, self.extra_read_dirs)

        self._namespace = {
            # Core data science
            "pd": pd,
            "np": np,
            "plt": plt,
            "json": json,
            "re": re,
            "math": math,
            "collections": collections,
            "itertools": itertools,
            "datetime": datetime,
            "zipfile": zipfile,
            "glob": glob_mod,
            "io": io,
            "tempfile": tempfile,
            "struct": struct,
            "csv": csv,
            "gzip": gzip,
            "os": os,
            "os.path": os.path,
            "urllib": __import__("urllib"),
            "urllib.request": urllib.request,
            "Path": Path,
            # Output directory
            "OUTPUT_DIR": self.output_dir,
            # Safe import
            "__builtins__": safe_builtins,
        }

        # Add safe_subprocess_run for whitelisted bioinformatics tools
        self._namespace["safe_subprocess_run"] = _make_safe_subprocess()

        # Optional libraries — add if available
        try:
            import scipy.stats as scipy_stats
            self._namespace["scipy_stats"] = scipy_stats
            self._namespace["scipy"] = __import__("scipy")
        except ImportError:
            pass

        try:
            import seaborn as sns
            self._namespace["sns"] = sns
        except ImportError:
            pass

        try:
            import sklearn
            self._namespace["sklearn"] = sklearn
        except ImportError:
            pass

        # rpy2 for R model fitting (when questions explicitly require R)
        try:
            import rpy2.robjects as ro
            self._namespace["ro"] = ro
            self._namespace["rpy2"] = __import__("rpy2")
        except ImportError:
            pass

        # Pre-built helper for parsimony informative sites (gap-correct)
        from collections import Counter as _Counter

        def compute_pi_percentage(seqs):
            """Compute parsimony informative site percentage. EXCLUDES gap characters.

            Only '-', 'X', and '*' are treated as gaps.
            'N' is NOT excluded (it's asparagine in protein alignments).
            '?' is NOT excluded (it may represent valid ambiguity).
            """
            if not seqs or len(seqs) < 2:
                return 0.0
            aln_len = len(seqs[0])
            pi_count = 0
            for i in range(aln_len):
                col = [s[i].upper() for s in seqs if i < len(s)]
                col = [c for c in col if c not in ('-', 'X', '*')]
                if len(col) < 2:
                    continue
                counts = _Counter(col)
                if sum(1 for c, n in counts.items() if n >= 2) >= 2:
                    pi_count += 1
            return pi_count / aln_len * 100

        self._namespace["compute_pi_percentage"] = compute_pi_percentage

    def load_datasets(self) -> dict:
        """Load configured datasets into the namespace. Returns dict of loaded names."""
        loaded = {}
        loaders = {
            "crispr": "load_crispr",
            "prism": "load_prism",
            "l1000": "load_l1000",
            "proteomics": "load_proteomics",
            "mutations": "load_mutations",
            "model_metadata": "load_model_metadata",
        }
        for name, func_name in loaders.items():
            try:
                from ct.data import loaders as data_loaders
                loader_fn = getattr(data_loaders, func_name, None)
                if loader_fn:
                    df = loader_fn()
                    self._namespace[name] = df
                    loaded[name] = f"DataFrame {df.shape[0]} rows x {df.shape[1]} cols"
            except (FileNotFoundError, Exception):
                pass  # Dataset not configured — skip silently
        return loaded

    def get_variable(self, name: str, default=None):
        """Retrieve a variable from the sandbox namespace."""
        return self._namespace.get(name, default)

    def inject_prior_results(self, prior_results: dict):
        """Add prior step results into namespace as step_1, step_2, etc."""
        if not prior_results:
            return
        for step_id, result in prior_results.items():
            self._namespace[f"step_{step_id}"] = result

    def describe_namespace(self) -> str:
        """Generate a text description of available data and libraries for the LLM."""
        lines = ["## Available in your namespace\n"]

        # Libraries
        libs = []
        lib_names = ["pd", "np", "plt", "sns", "scipy_stats", "scipy", "sklearn",
                     "json", "re", "math", "collections", "itertools", "datetime",
                     "ro"]
        for name in lib_names:
            if name in self._namespace:
                libs.append(name)
        lines.append(f"**Libraries**: {', '.join(libs)}")
        lines.append(f"**OUTPUT_DIR**: Path('{self.output_dir}') — save plots/CSVs here\n")

        # Datasets
        dataset_names = ["crispr", "prism", "l1000", "proteomics", "mutations", "model_metadata"]
        for name in dataset_names:
            if name in self._namespace:
                df = self._namespace[name]
                cols_preview = list(df.columns[:8])
                if len(df.columns) > 8:
                    cols_preview.append(f"... ({len(df.columns)} total)")
                lines.append(
                    f"**{name}**: DataFrame({df.shape[0]} rows x {df.shape[1]} cols), "
                    f"columns: {cols_preview}, index: {df.index.name or type(df.index).__name__}"
                )

        # Pre-built helpers
        helpers = []
        if "compute_pi_percentage" in self._namespace:
            helpers.append("compute_pi_percentage(seqs) — computes parsimony informative site % with gap exclusion")
        if "safe_subprocess_run" in self._namespace:
            helpers.append("safe_subprocess_run(cmd) — run whitelisted bioinformatics CLI tools")
        if helpers:
            lines.append("**Pre-imported helper functions** (already available — just call them):")
            for h in helpers:
                lines.append(f"  - `{h}`")
            lines.append("")

        # Prior results
        steps = [k for k in self._namespace if k.startswith("step_")]
        if steps:
            lines.append(f"\n**Prior step results**: {', '.join(sorted(steps))}")
            for s in sorted(steps):
                val = self._namespace[s]
                if isinstance(val, dict):
                    keys = list(val.keys())[:6]
                    lines.append(f"  {s}: dict with keys {keys}")
                else:
                    lines.append(f"  {s}: {type(val).__name__}")

        return "\n".join(lines)

    def _protect_preimported_helpers(self, code: str) -> str:
        """Remove any user redefinition of pre-imported helper functions.

        The sandbox pre-imports correct versions of helpers like compute_pi_percentage.
        LLM-generated code sometimes redefines these incorrectly (e.g., including gap
        characters in PI computation). This method strips such redefinitions so the
        pre-imported versions are used instead.
        """
        import ast

        protected = {"compute_pi_percentage"}
        # Only protect functions that are actually in the namespace
        protected = {f for f in protected if f in self._namespace}
        if not protected:
            return code

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # Can't parse — let execution handle the error

        # Find line ranges of protected function definitions
        lines = code.split('\n')
        remove_ranges = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in protected:
                start = node.lineno - 1  # 0-indexed
                end = node.end_lineno  # end_lineno is 1-indexed, exclusive after -1+1
                remove_ranges.append((start, end))

        if not remove_ranges:
            return code

        # Remove the function definitions (replace with comment)
        new_lines = []
        skip_until = -1
        for i, line in enumerate(lines):
            if i < skip_until:
                continue
            removed = False
            for start, end in remove_ranges:
                if i == start:
                    new_lines.append(f'# {line.strip()} — REMOVED: using pre-imported sandbox version')
                    skip_until = end
                    removed = True
                    break
            if not removed and i >= skip_until:
                new_lines.append(line)

        return '\n'.join(new_lines)

    def execute(self, code: str) -> dict:
        """Execute code in the sandbox and return results.

        Returns dict with: stdout, stderr, result, error, plots, exports.
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot output dir contents before execution
        existing_files = set(self.output_dir.iterdir()) if self.output_dir.exists() else set()

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()

        result = {
            "stdout": "",
            "stderr": "",
            "result": None,
            "error": None,
            "plots": [],
            "exports": [],
        }

        # Set up timeout (Unix only)
        import threading
        has_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
        if has_alarm:
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution timed out after {self.timeout}s")
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.timeout)

        try:
            sys.stdout = captured_out
            sys.stderr = captured_err

            # Strip any redefinition of pre-imported helpers to ensure sandbox versions are used
            code = self._protect_preimported_helpers(code)

            # Compile and execute
            compiled = compile(code, "<ct-sandbox>", "exec")
            exec(compiled, self._namespace)

            # Capture the 'result' variable if set by the code
            if "result" in self._namespace:
                result["result"] = self._namespace["result"]

        except TimeoutError as e:
            result["error"] = str(e)
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            # Restore
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        result["stdout"] = captured_out.getvalue()
        result["stderr"] = captured_err.getvalue()

        # Detect new files in output dir
        if self.output_dir.exists():
            new_files = set(self.output_dir.iterdir()) - existing_files
            for f in sorted(new_files):
                if f.suffix in (".png", ".svg", ".jpg", ".jpeg", ".pdf"):
                    result["plots"].append(str(f))
                elif f.suffix in (".csv", ".xlsx", ".tsv", ".json"):
                    result["exports"].append(str(f))

        return result
