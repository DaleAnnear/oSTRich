"""Create plots and a plain-English report for an oSTRich benchmark run."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ostrich_rnn.benchmark_report import main


if __name__ == "__main__":
    main()
