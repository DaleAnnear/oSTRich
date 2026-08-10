"""Download and prepare TRExplorer variation clusters as an oSTRich truth set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ostrich_rnn.truth_sets import extract_variation_clusters, summarize_trexplorer_bed, write_summary


TREXPLORER_V2_ASSET = (
    "https://github.com/broadinstitute/trexplorer-catalog/releases/download/v2.0/"
    "TRExplorer.variation_clusters_and_isolated_TRs_v2.hg38.TRGT.bed.gz"
)
TREXPLORER_V2_FILENAME = "TRExplorer.variation_clusters_and_isolated_TRs_v2.hg38.TRGT.bed.gz"
TREXPLORER_V2_CLUSTERS_FILENAME = "TRExplorer.variation_clusters_v2.hg38.TRGT.bed.gz"


def download(url: str, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, output_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="truth_sets/trexplorer_v2",
        help="Directory for the downloaded BED, clusters-only BED, and summary JSON.",
    )
    parser.add_argument("--url", default=TREXPLORER_V2_ASSET, help="TRExplorer BED.gz asset URL.")
    parser.add_argument("--force", action="store_true", help="Redownload the upstream BED.gz if it already exists.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download and summarize the upstream BED.gz; do not write a clusters-only BED.gz.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    raw_bed = output_dir / TREXPLORER_V2_FILENAME
    clusters_bed = output_dir / TREXPLORER_V2_CLUSTERS_FILENAME
    summary_path = output_dir / "truth_set_summary.json"

    download(args.url, raw_bed, force=args.force)
    summary = summarize_trexplorer_bed(raw_bed)
    summary.update(
        {
            "source_url": args.url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_bed": str(raw_bed),
        }
    )

    if not args.skip_extract:
        cluster_count = extract_variation_clusters(raw_bed, clusters_bed)
        summary["clusters_bed"] = str(clusters_bed)
        summary["clusters_bed_records"] = cluster_count

    write_summary(summary_path, summary)
    print(f"raw_bed={raw_bed}")
    if "clusters_bed" in summary:
        print(f"clusters_bed={summary['clusters_bed']}")
    print(f"summary={summary_path}")
    print(f"total_records={summary['total_records']}")
    print(f"kind_counts={summary['kind_counts']}")
    return summary


if __name__ == "__main__":
    main()
