"""Minimal Python-to-R bridge.

R earns its place here for one thing: hierarchical models with partial pooling,
where lme4 is the reference implementation and statsmodels MixedLM is fragile
with random slopes. Everything else stays in Python.

Data crosses the boundary as CSV. That is deliberately the laziest workable
choice: the R arrow package is a heavy compile, and these frames are small.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

USER_LIB = "C:/Users/Max/R/win-library/4.6"


def rscript_path() -> str:
    """Locate Rscript.exe, preferring the newest install."""
    cands = sorted(Path("C:/Program Files/R").glob("R-*/bin/x64/Rscript.exe"))
    if not cands:
        cands = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
    if not cands:
        raise RuntimeError("Rscript.exe not found under C:/Program Files/R")
    return str(cands[-1])


def run_r(script: str | Path, args: list[str] | None = None,
          timeout: int = 900) -> str:
    """Run an R script and return its stdout. Raises on non-zero exit."""
    cmd = [rscript_path(), "--vanilla", str(script)] + [str(a) for a in (args or [])]
    env_line = f'.libPaths(c("{USER_LIB}", .libPaths()))'
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**__import__("os").environ,
                               "R_LIBS_USER": USER_LIB,
                               "R_PROFILE_USER": ""})
    if proc.returncode != 0:
        raise RuntimeError(f"R failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return proc.stdout
