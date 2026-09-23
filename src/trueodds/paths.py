"""Where data and runs live (D12). Override the root with TRUEODDS_HOME."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("TRUEODDS_HOME", Path.home() / ".trueodds"))
DATA = HOME / "data"
RAW = DATA / "raw"  # the Hugging Face download cache
RUNS = HOME / "runs"
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "harness" / "results"
