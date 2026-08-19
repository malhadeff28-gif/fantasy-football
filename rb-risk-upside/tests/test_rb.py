"""One command that fails if anything load-bearing breaks.

Each module owns its own check; this just runs all three. The checks that
matter are the as-of guarantee in rb_features and the no-look-ahead guarantee
in rb_simulate - everything else in this project is downstream of those two
being true.

    ./.venv/Scripts/python.exe rb-risk-upside/tests/test_rb.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rb_data
import rb_features
import rb_simulate


def test_all() -> None:
    for mod in (rb_data, rb_features, rb_simulate):
        print(f"--- {mod.__name__}")
        mod._demo()


if __name__ == "__main__":
    test_all()
    print("\nall rb-risk-upside checks passed")
