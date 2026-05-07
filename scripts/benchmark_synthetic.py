"""Run an automated synthetic oSTRich benchmark sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ostrich_rnn.benchmark import default_benchmark_config, load_benchmark_config, run_benchmark, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or create an oSTRich benchmark configuration.")
    parser.add_argument("--config", default=None, help="Benchmark JSON config. Uses a small default sweep if omitted.")
    parser.add_argument("--output-root", default=None, help="Override the benchmark output root directory.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None, help="Override the config device.")
    parser.add_argument("--write-template", default=None, help="Write an editable benchmark JSON template and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_json(args.write_template, default_benchmark_config())
        print(f"template={args.write_template}")
        return

    config = load_benchmark_config(args.config) if args.config else default_benchmark_config()
    result = run_benchmark(config, output_root=args.output_root, device=args.device)
    print(f"benchmark_dir={result['benchmark_dir']}")
    print(f"comparison_csv={result['comparison_csv']}")
    print(f"comparison_json={result['comparison_json']}")


if __name__ == "__main__":
    main()
