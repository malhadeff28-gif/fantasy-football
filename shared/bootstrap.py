"""Put the shared module folders on sys.path.

The shared folders are hyphenated per the repository layout, which makes them
invalid package names, so they are added to sys.path directly and their modules
imported flat. One helper beats a naming convention nobody remembers.

Usage from any project script:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import bootstrap; bootstrap.setup()
    import nfl_data, markets, scoring, evaluation
"""
from __future__ import annotations

import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent
REPO = SHARED.parent
CACHE = SHARED / "nfl-data" / "cache"


def setup() -> Path:
    """Add every shared module folder to sys.path. Idempotent."""
    for name in ("nfl-data", "market-data", "utilities"):
        p = str(SHARED / name)
        if p not in sys.path:
            sys.path.insert(0, p)
    return REPO
