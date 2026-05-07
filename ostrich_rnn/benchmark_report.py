"""Interpret benchmark comparison tables and create summary plots."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any


PRIMARY_METRICS = (
    "holdout_tract_f1",
    "holdout_base_f1",
    "holdout_motif_accuracy",
    "holdout_motif_length_accuracy",
    "holdout_tract_motif_accuracy",
)

LOWER_IS_BETTER = ("holdout_false_positives", "holdout_false_negatives", "holdout_length_mae")
FACTOR_COLUMNS = ("batch_size", "hidden_dim", "num_layers", "epochs", "curriculum")


@dataclass
class BenchmarkReport:
    benchmark_dir: str
    report_dir: str
    summary_path: str
    html_path: str
    ranked_csv: str
    plot_paths: list[str]
    best_experiment: str | None


def parse_value(value: str) -> Any:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_comparison(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return [row for row in rows if row.get("status") in (None, "completed")]


def metric_value(row: dict, metric: str) -> float | None:
    value = row.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def available_metrics(rows: list[dict]) -> list[str]:
    return [metric for metric in PRIMARY_METRICS if any(metric_value(row, metric) is not None for row in rows)]


def rank_rows(rows: list[dict], primary_metric: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            metric_value(row, primary_metric) is None,
            -(metric_value(row, primary_metric) or float("-inf")),
            -(metric_value(row, "holdout_motif_accuracy") or float("-inf")),
            metric_value(row, "holdout_false_positives") or float("inf"),
        ),
    )


def write_ranked_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field_order = [
        "rank",
        "experiment",
        "holdout_tract_f1",
        "holdout_base_f1",
        "holdout_motif_accuracy",
        "holdout_motif_length_accuracy",
        "holdout_tract_motif_accuracy",
        "holdout_false_positives",
        "holdout_false_negatives",
        "holdout_length_mae",
        "batch_size",
        "hidden_dim",
        "num_layers",
        "epochs",
        "curriculum",
        "best_val_loss",
        "best_epoch",
        "epochs_run",
        "best_checkpoint",
    ]
    extra_fields = sorted({key for row in rows for key in row} - set(field_order))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_order + extra_fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})


def factor_summary(rows: list[dict], factor: str, metric: str) -> list[tuple[Any, float, int]]:
    groups: dict[Any, list[float]] = {}
    for row in rows:
        if factor not in row:
            continue
        value = metric_value(row, metric)
        if value is None:
            continue
        groups.setdefault(row[factor], []).append(value)
    return sorted(
        [(key, mean(values), len(values)) for key, values in groups.items()],
        key=lambda item: str(item[0]),
    )


def best_factor_lines(rows: list[dict], metric: str) -> list[str]:
    lines = []
    for factor in FACTOR_COLUMNS:
        summary = factor_summary(rows, factor, metric)
        if len(summary) < 2:
            continue
        best = max(summary, key=lambda item: item[1])
        worst = min(summary, key=lambda item: item[1])
        delta = best[1] - worst[1]
        lines.append(
            f"- {factor}: best average {metric} was {best[1]:.4f} for `{best[0]}` "
            f"(n={best[2]}), {delta:.4f} above the lowest setting."
        )
    return lines


def concise_setting(row: dict) -> str:
    return (
        f"batch_size={row.get('batch_size')}, hidden_dim={row.get('hidden_dim')}, "
        f"num_layers={row.get('num_layers')}, epochs={row.get('epochs')}, "
        f"curriculum={row.get('curriculum')}"
    )


def write_markdown_summary(path: str | Path, rows: list[dict], ranked: list[dict], primary_metric: str) -> None:
    path = Path(path)
    best = ranked[0] if ranked else None
    lines = [
        "# oSTRich Benchmark Interpretation",
        "",
        f"Compared completed runs: {len(rows)}",
        f"Primary ranking metric: `{primary_metric}`",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Recommended Setting",
                "",
                f"Best run: `{best.get('experiment')}`",
                "",
                f"Settings: `{concise_setting(best)}`",
                "",
                "Key holdout metrics:",
                "",
                f"- holdout tract F1: {metric_value(best, 'holdout_tract_f1') or 0.0:.4f}",
                f"- holdout base F1: {metric_value(best, 'holdout_base_f1') or 0.0:.4f}",
                f"- holdout motif accuracy: {metric_value(best, 'holdout_motif_accuracy') or 0.0:.4f}",
                f"- holdout motif-length accuracy: {metric_value(best, 'holdout_motif_length_accuracy') or 0.0:.4f}",
                f"- holdout tract motif accuracy: {metric_value(best, 'holdout_tract_motif_accuracy') or 0.0:.4f}",
                "",
            ]
        )
    lines.extend(["## Top Runs", ""])
    for rank, row in enumerate(ranked[:10], start=1):
        value = metric_value(row, primary_metric)
        lines.append(
            f"{rank}. `{row.get('experiment')}`: {primary_metric}={value:.4f} "
            f"({concise_setting(row)})"
        )
    lines.extend(["", "## Factor Effects", ""])
    factor_lines = best_factor_lines(rows, primary_metric)
    lines.extend(factor_lines or ["No factor effects could be estimated from this table."])
    lines.extend(
        [
            "",
            "## How To Use This",
            "",
            "Use the top-ranked setting as the leading candidate, but prefer a simpler model when scores are close.",
            "For example, if a one-layer model is within about 1-2 percentage points of a two-layer model, the one-layer model is usually the better next-round choice because it trains faster and is easier to tune.",
            "Treat this report as a pilot result: rerun the best few settings on a larger and more realistic benchmark before making them the project default.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table_columns(rows: list[dict]) -> list[str]:
    preferred = [
        "rank",
        "experiment",
        "holdout_tract_f1",
        "holdout_base_f1",
        "holdout_motif_accuracy",
        "holdout_motif_length_accuracy",
        "holdout_tract_motif_accuracy",
        "holdout_false_positives",
        "holdout_false_negatives",
        "holdout_length_mae",
        "batch_size",
        "hidden_dim",
        "num_layers",
        "epochs",
        "curriculum",
        "best_val_loss",
        "best_epoch",
        "epochs_run",
        "best_checkpoint",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + extras


def html_metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(display_value(value))}</div>'
        "</div>"
    )


def html_results_table(rows: list[dict]) -> str:
    columns = table_columns(rows)
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows = []
    for rank, row in enumerate(rows, start=1):
        ranked_row = {"rank": rank, **row}
        cells = "".join(f"<td>{escape(display_value(ranked_row.get(column)))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap">'
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def write_html_report(
    path: str | Path,
    rows: list[dict],
    ranked: list[dict],
    primary_metric: str,
    plot_paths: list[str],
    ranked_csv: str | Path,
    summary_path: str | Path,
) -> None:
    path = Path(path)
    best = ranked[0] if ranked else None
    factor_items = "".join(f"<li>{escape(line.lstrip('- '))}</li>" for line in best_factor_lines(rows, primary_metric))
    if not factor_items:
        factor_items = "<li>No factor effects could be estimated from this table.</li>"
    plot_cards = []
    for plot_path in plot_paths:
        rel_path = Path(plot_path).relative_to(path.parent)
        title = Path(plot_path).stem.replace("_", " ").title()
        plot_cards.append(
            '<section class="plot-card">'
            f"<h3>{escape(title)}</h3>"
            f'<img src="{escape(str(rel_path).replace(chr(92), "/"))}" alt="{escape(title)}">'
            "</section>"
        )
    if not plot_cards:
        plot_cards.append('<p class="muted">No plots were generated for this benchmark table.</p>')

    best_section = ""
    if best:
        best_section = f"""
        <section class="panel">
          <h2>Recommended Setting</h2>
          <p class="lead">Best run: <strong>{escape(str(best.get('experiment')))}</strong></p>
          <p><code>{escape(concise_setting(best))}</code></p>
          <div class="metric-grid">
            {html_metric_card("Holdout tract F1", metric_value(best, "holdout_tract_f1"))}
            {html_metric_card("Holdout base F1", metric_value(best, "holdout_base_f1"))}
            {html_metric_card("Motif accuracy", metric_value(best, "holdout_motif_accuracy"))}
            {html_metric_card("Motif-length accuracy", metric_value(best, "holdout_motif_length_accuracy"))}
            {html_metric_card("Tract motif accuracy", metric_value(best, "holdout_tract_motif_accuracy"))}
          </div>
        </section>
        """

    top_runs = []
    for rank, row in enumerate(ranked[:10], start=1):
        value = metric_value(row, primary_metric)
        top_runs.append(
            "<li>"
            f"<strong>{rank}. {escape(str(row.get('experiment')))}</strong>: "
            f"{escape(primary_metric)}={escape(display_value(value))} "
            f"<span class=\"muted\">({escape(concise_setting(row))})</span>"
            "</li>"
        )

    ranked_csv_rel = Path(ranked_csv).relative_to(path.parent)
    summary_rel = Path(summary_path).relative_to(path.parent)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>oSTRich Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #627184;
      --line: #d9e0e7;
      --accent: #2f7f73;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    header {{
      padding: 32px 40px 22px;
      background: #102027;
      color: white;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
    }}
    main {{
      padding: 24px 40px 42px;
      max-width: 1320px;
      margin: 0 auto;
    }}
    .panel, .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    .lead {{
      font-size: 1.08rem;
    }}
    .muted {{
      color: var(--muted);
    }}
    code {{
      background: #eef2f5;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    a {{
      color: var(--accent);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 5px;
    }}
    .metric-value {{
      font-size: 1.35rem;
      font-weight: 700;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      align-items: start;
    }}
    .plot-card img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.88rem;
      white-space: nowrap;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eaf0f4;
      z-index: 1;
    }}
    tbody tr:nth-child(even) {{
      background: #fafbfc;
    }}
  </style>
</head>
<body>
  <header>
    <h1>oSTRich Benchmark Report</h1>
    <p>Compared {len(rows)} completed runs. Primary ranking metric: <code>{escape(primary_metric)}</code>.</p>
  </header>
  <main>
    {best_section}
    <section class="panel">
      <h2>Top Runs</h2>
      <ol>{''.join(top_runs)}</ol>
    </section>
    <section class="panel">
      <h2>Factor Effects</h2>
      <ul>{factor_items}</ul>
    </section>
    <section class="panel">
      <h2>Report Files</h2>
      <p><a href="{escape(str(ranked_csv_rel).replace(chr(92), "/"))}">Ranked results CSV</a></p>
      <p><a href="{escape(str(summary_rel).replace(chr(92), "/"))}">Markdown summary</a></p>
    </section>
    <section>
      <h2>Plots</h2>
      <div class="plot-grid">{''.join(plot_cards)}</div>
    </section>
    <section class="panel">
      <h2>All Ranked Results</h2>
      {html_results_table(ranked)}
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for benchmark plots. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc


def save_top_runs_plot(rows: list[dict], metric: str, output_path: Path) -> str | None:
    ranked = rank_rows(rows, metric)[:10]
    if not ranked:
        return None
    plt = import_pyplot()
    labels = [str(row.get("experiment")) for row in ranked]
    values = [metric_value(row, metric) or 0.0 for row in ranked]
    fig_height = max(4.0, 0.42 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = list(range(len(labels)))
    ax.barh(y_positions, values, color="#2f7f73")
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title("Top Benchmark Runs")
    ax.set_xlim(0, max(1.0, max(values) * 1.08 if values else 1.0))
    for y, value in zip(y_positions, values):
        ax.text(value + 0.01, y, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def save_factor_effects_plot(rows: list[dict], metric: str, output_path: Path) -> str | None:
    summaries = [(factor, factor_summary(rows, factor, metric)) for factor in FACTOR_COLUMNS]
    summaries = [(factor, summary) for factor, summary in summaries if len(summary) >= 2]
    if not summaries:
        return None
    plt = import_pyplot()
    fig, axes = plt.subplots(len(summaries), 1, figsize=(9, max(3, 2.4 * len(summaries))))
    if len(summaries) == 1:
        axes = [axes]
    for ax, (factor, summary) in zip(axes, summaries):
        labels = [str(item[0]) for item in summary]
        values = [item[1] for item in summary]
        ax.bar(labels, values, color="#4b78a8")
        ax.set_ylabel(metric)
        ax.set_title(f"Average {metric} by {factor}")
        ax.set_ylim(0, max(1.0, max(values) * 1.08 if values else 1.0))
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def pivot_grid(rows: list[dict], x_key: str, y_key: str, metric: str, filters: dict) -> tuple[list[Any], list[Any], list[list[float | None]]]:
    filtered = []
    for row in rows:
        keep = True
        for key, expected in filters.items():
            if row.get(key) != expected:
                keep = False
                break
        if keep and metric_value(row, metric) is not None:
            filtered.append(row)
    x_values = sorted({row.get(x_key) for row in filtered}, key=lambda value: str(value))
    y_values = sorted({row.get(y_key) for row in filtered}, key=lambda value: str(value))
    grid = []
    for y in y_values:
        row_values = []
        for x in x_values:
            values = [metric_value(row, metric) for row in filtered if row.get(x_key) == x and row.get(y_key) == y]
            clean = [value for value in values if value is not None]
            row_values.append(mean(clean) if clean else None)
        grid.append(row_values)
    return x_values, y_values, grid


def save_hidden_batch_heatmaps(rows: list[dict], metric: str, output_path: Path) -> str | None:
    if not all(any(key in row for row in rows) for key in ("hidden_dim", "batch_size", "curriculum")):
        return None
    panels = []
    for curriculum in (False, True):
        x_values, y_values, grid = pivot_grid(
            rows,
            x_key="hidden_dim",
            y_key="batch_size",
            metric=metric,
            filters={"curriculum": curriculum},
        )
        if x_values and y_values:
            panels.append((curriculum, x_values, y_values, grid))
    if not panels:
        return None
    plt = import_pyplot()
    fig_width = 5.8 * len(panels) + 0.7
    fig = plt.figure(figsize=(fig_width, 4.4), constrained_layout=True)
    grid_spec = fig.add_gridspec(1, len(panels) + 1, width_ratios=[1] * len(panels) + [0.055])
    axes = [fig.add_subplot(grid_spec[0, idx]) for idx in range(len(panels))]
    colorbar_axis = fig.add_subplot(grid_spec[0, len(panels)])
    image = None
    for ax, (curriculum, x_values, y_values, grid) in zip(axes, panels):
        numeric = [[value if value is not None else float("nan") for value in row] for row in grid]
        image = ax.imshow(numeric, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(x_values)), [str(value) for value in x_values])
        ax.set_yticks(range(len(y_values)), [str(value) for value in y_values])
        ax.set_xlabel("hidden_dim")
        ax.set_ylabel("batch_size")
        ax.set_title(f"{metric}, curriculum={curriculum}")
        for y_idx, row in enumerate(grid):
            for x_idx, value in enumerate(row):
                label = "NA" if value is None else f"{value:.3f}"
                ax.text(x_idx, y_idx, label, ha="center", va="center", color="white", fontsize=9)
    if image is not None:
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(metric, rotation=270, labelpad=18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def create_plots(rows: list[dict], metric: str, report_dir: Path) -> list[str]:
    plot_paths = []
    for maybe_path in (
        save_top_runs_plot(rows, metric, report_dir / "top_runs.png"),
        save_factor_effects_plot(rows, metric, report_dir / "factor_effects.png"),
        save_hidden_batch_heatmaps(rows, metric, report_dir / "hidden_dim_by_batch_size.png"),
    ):
        if maybe_path:
            plot_paths.append(maybe_path)
    return plot_paths


def create_benchmark_report(
    benchmark_dir: str | Path,
    metric: str = "holdout_tract_f1",
    output_dir: str | Path | None = None,
) -> BenchmarkReport:
    benchmark_dir = Path(benchmark_dir)
    comparison_path = benchmark_dir / "comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Could not find benchmark comparison file: {comparison_path}")
    rows = read_comparison(comparison_path)
    if not rows:
        raise ValueError(f"No completed benchmark rows found in {comparison_path}")
    metrics = available_metrics(rows)
    if metric not in metrics:
        metric = metrics[0] if metrics else "best_tract_f1"
    ranked = rank_rows(rows, metric)
    report_dir = Path(output_dir) if output_dir else benchmark_dir / "report"
    ranked_csv = report_dir / "ranked_results.csv"
    summary_path = report_dir / "summary.md"
    html_path = report_dir / "report.html"
    write_ranked_csv(ranked_csv, ranked)
    write_markdown_summary(summary_path, rows, ranked, metric)
    plot_paths = create_plots(rows, metric, report_dir)
    write_html_report(html_path, rows, ranked, metric, plot_paths, ranked_csv, summary_path)
    return BenchmarkReport(
        benchmark_dir=str(benchmark_dir),
        report_dir=str(report_dir),
        summary_path=str(summary_path),
        html_path=str(html_path),
        ranked_csv=str(ranked_csv),
        plot_paths=plot_paths,
        best_experiment=str(ranked[0].get("experiment")) if ranked else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots and a summary from an oSTRich benchmark run.")
    parser.add_argument("benchmark_dir", help="Benchmark directory containing comparison.csv.")
    parser.add_argument("--metric", default="holdout_tract_f1", help="Metric used to rank and plot runs.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to benchmark_dir/report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_benchmark_report(args.benchmark_dir, metric=args.metric, output_dir=args.output_dir)
    print(f"report_dir={report.report_dir}")
    print(f"html={report.html_path}")
    print(f"summary={report.summary_path}")
    print(f"ranked_csv={report.ranked_csv}")
    for path in report.plot_paths:
        print(f"plot={path}")
    if report.best_experiment:
        print(f"best_experiment={report.best_experiment}")


if __name__ == "__main__":
    main()
