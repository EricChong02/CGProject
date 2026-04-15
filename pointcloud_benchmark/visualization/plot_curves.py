"""Training curve plotting helpers."""

from __future__ import annotations

from pathlib import Path

def plot_training_history(history: dict, output_path: Path, title: str, dpi: int = 150) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 900
    height = 540
    margin_left = 70
    margin_right = 30
    margin_top = 60
    margin_bottom = 70

    series = [
        ("Train Loss", history.get("train_loss", []), "#d1495b"),
        ("Train Accuracy", history.get("train_acc", []), "#00798c"),
        ("Validation Accuracy", history.get("val_acc", []), "#edae49"),
    ]
    max_len = max((len(values) for _, values, _ in series), default=0)
    max_value = max((max(values) for _, values, _ in series if values), default=1.0)
    min_value = min((min(values) for _, values, _ in series if values), default=0.0)
    value_range = max(max_value - min_value, 1e-6)

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_x(index: int) -> float:
        if max_len <= 1:
            return margin_left + plot_width / 2
        return margin_left + (index / (max_len - 1)) * plot_width

    def scale_y(value: float) -> float:
        normalized = (value - min_value) / value_range
        return margin_top + plot_height - normalized * plot_height

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8" />',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#444" stroke-width="2" />',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#444" stroke-width="2" />',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16" font-family="Arial">Epoch</text>',
        f'<text x="24" y="{height / 2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 24 {height / 2})">Value</text>',
    ]

    for tick in range(5):
        value = min_value + (value_range * tick / 4)
        y = scale_y(value)
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#ddd" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial">{value:.2f}</text>'
        )

    for epoch_idx in range(max_len):
        x = scale_x(epoch_idx)
        svg_lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#f0f0f0" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial">{epoch_idx + 1}</text>'
        )

    legend_x = width - margin_right - 200
    legend_y = margin_top - 20
    for idx, (label, values, color) in enumerate(series):
        if values:
            points = " ".join(
                f"{scale_x(i):.2f},{scale_y(value):.2f}" for i, value in enumerate(values)
            )
            svg_lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />'
            )
            for i, value in enumerate(values):
                svg_lines.append(
                    f'<circle cx="{scale_x(i):.2f}" cy="{scale_y(value):.2f}" r="3.5" fill="{color}" />'
                )

        y = legend_y + idx * 22
        svg_lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3" />'
        )
        svg_lines.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-size="13" font-family="Arial">{label}</text>'
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")

    # TODO: Add support for confusion matrices, class histograms, and point-cloud snapshots.
