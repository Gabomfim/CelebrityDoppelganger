"""Plot validation cluster geometry over training epochs from W&B runs."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import wandb

METRICS = (
    ("validation/intra_class_distance", "Intra-class distance", "lower is better"),
    ("validation/nearest_centroid_distance", "Nearest-centroid distance", "higher is better"),
    ("validation/separation_ratio", "Separation ratio", "higher is better"),
)
COLORS = ("#A50F35", "#D1962E", "#59477A", "#E36B4F", "#287271")


def fetch_histories(entity: str, project: str, run_names: list[str]) -> dict[str, list[dict]]:
    api = wandb.Api()
    runs = {run.name: run for run in api.runs(f"{entity}/{project}") if run.name in run_names}
    missing = sorted(set(run_names) - set(runs))
    if missing:
        raise ValueError(f"W&B runs not found: {', '.join(missing)}")
    keys = ["epoch", *(metric[0] for metric in METRICS)]
    return {name: runs[name].history(keys=keys, pandas=False, samples=10_000) for name in run_names}


def scale(value: float, low: float, high: float, start: float, length: float) -> float:
    if high == low:
        return start + length / 2
    return start + (value - low) / (high - low) * length


def render_svg(histories: dict[str, list[dict]], output: Path) -> None:
    width, height = 1500, 1110
    margin_x, plot_width = 110, 1280
    panel_height, panel_gap = 220, 70
    top = 180
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFF8ED"/>',
        '<text x="75" y="72" font-family="Arial" font-size="38" font-weight="700" '
        'fill="#290D15">Cluster geometry during training</text>',
        '<text x="75" y="110" font-family="Arial" font-size="19" fill="#765D63">'
        "Cross-entropy baseline vs supervised contrastive candidates · validation split</text>",
    ]

    for panel, (key, title, direction) in enumerate(METRICS):
        y0 = top + panel * (panel_height + panel_gap)
        series = {
            name: [(int(row["epoch"]), float(row[key])) for row in rows if key in row]
            for name, rows in histories.items()
        }
        values = [value for points in series.values() for _, value in points]
        epochs = [epoch for points in series.values() for epoch, _ in points]
        if not values:
            continue
        low, high = min(values), max(values)
        padding = max((high - low) * 0.08, 1e-4)
        low, high = low - padding, high + padding
        first_epoch, last_epoch = min(epochs), max(epochs)

        parts.extend(
            [
                f'<text x="{margin_x}" y="{y0 - 18}" font-family="Arial" font-size="24" '
                f'font-weight="700" fill="#290D15">{html.escape(title)}</text>',
                f'<text x="{margin_x + 330}" y="{y0 - 18}" font-family="Arial" font-size="16" '
                f'fill="#A50F35">{direction}</text>',
                f'<line x1="{margin_x}" y1="{y0}" x2="{margin_x}" y2="{y0 + panel_height}" '
                'stroke="#D8C9BE" stroke-width="2"/>',
                f'<line x1="{margin_x}" y1="{y0 + panel_height}" '
                f'x2="{margin_x + plot_width}" y2="{y0 + panel_height}" '
                'stroke="#D8C9BE" stroke-width="2"/>',
                f'<text x="{margin_x - 15}" y="{y0 + 8}" text-anchor="end" '
                f'font-family="Arial" font-size="14" fill="#765D63">{high:.3f}</text>',
                f'<text x="{margin_x - 15}" y="{y0 + panel_height + 5}" text-anchor="end" '
                f'font-family="Arial" font-size="14" fill="#765D63">{low:.3f}</text>',
            ]
        )
        for index, (_name, points) in enumerate(series.items()):
            if not points:
                continue
            color = COLORS[index % len(COLORS)]
            coords = []
            for epoch, value in points:
                x = scale(epoch, first_epoch, last_epoch, margin_x, plot_width)
                y = y0 + panel_height - scale(value, low, high, 0, panel_height)
                coords.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" '
                'stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>'
            )

    legend_y = height - 35
    for index, name in enumerate(histories):
        x = 80 + index * 330
        color = COLORS[index % len(COLORS)]
        parts.extend(
            [
                f'<line x1="{x}" y1="{legend_y}" x2="{x + 45}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="7" stroke-linecap="round"/>',
                f'<text x="{x + 60}" y="{legend_y + 6}" font-family="Arial" font-size="15" '
                f'fill="#290D15">{html.escape(name)}</text>',
            ]
        )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("cluster_geometry.svg"))
    args = parser.parse_args()
    render_svg(fetch_histories(args.entity, args.project, args.runs), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
