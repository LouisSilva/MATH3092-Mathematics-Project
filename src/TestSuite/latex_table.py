import re
from typing import Iterable

import pandas as pd

from .metrics import (
    Metric,
    TruePositiveRate,
    FalsePositiveRate,
    Precision,
    MeanQueryTime,
)

DEFAULT_TABLE_METRICS: list[Metric] = [
    TruePositiveRate(),
    FalsePositiveRate(),
    Precision(),
    MeanQueryTime(),
]

DEFAULT_METRIC_HEADERS: dict[str, str] = {
    "recall_pct": r"\textbf{Recall}(\%)",
    "false_pos_rate_pct": r"\textbf{FPR}(\%)",
    "precision_pct": r"\textbf{Precision}(\%)",
    "mean_query_time_s": r"\textbf{Mean Query Time}(s)",
}

DEFAULT_METRIC_FORMAT: dict[str, str] = {
    "recall_pct": ".1f",
    "false_pos_rate_pct": ".1f",
    "precision_pct": ".1f",
    "mean_query_time_s": ".3f",
}

PREAMBLE = r"""% Required: \usepackage{xcolor}, \usepackage{booktabs}, \usepackage{tikz}
\definecolor{pcacolor}{RGB}{230, 120, 30}    % orange
\definecolor{shazamcolor}{RGB}{70, 170, 230} % blue

\newcommand{\splitcell}[2]{%
  \begin{tikzpicture}[baseline=(current bounding box.center)]
    \def\w{1.7}\def\h{0.9}
    \fill[pcacolor!70]    (0,\h) -- (\w,\h) -- (0,0) -- cycle;
    \fill[shazamcolor!70] (\w,\h) -- (\w,0)  -- (0,0) -- cycle;
    \draw[black] (0,0) rectangle (\w,\h);
    \draw[black] (\w,\h) -- (0,0);
    \node[anchor=north west, font=\small, inner sep=2pt] at (0,\h) {#1};
    \node[anchor=south east, font=\small, inner sep=2pt] at (\w,0) {#2};
  \end{tikzpicture}%
}
"""


def _compute_per_cell_values(
        results_df: pd.DataFrame,
        metrics: list[Metric],
) -> dict[tuple[str, str], dict[str, float | None]]:
    """Returns {(backend, test_case): {metric_name: value}}."""
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    for (backend, test_case), group in results_df.groupby(["Backend", "Test Case"]):
        out[(str(backend), str(test_case))] = {
            metric.name: metric.compute(group) for metric in metrics
        }
    return out


def _format_cell(value, fmt: str, missing: str = "--") -> str:
    if value is None or pd.isna(value):
        return missing
    return f"{value:{fmt}}"


def _latex_escape(s: str) -> str:
    """Minimal escape for characters that would break LaTeX."""
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


_NUMBER = r"(-?\d+(?:\.\d+)?)"


def _default_pretty_name(test_case_str: str) -> str:
    if test_case_str == "CleanTrackTest()":
        return "Clean Track"

    match = re.fullmatch(rf"WhiteNoiseTest\(snr_db={_NUMBER}\)", test_case_str)
    if match:
        return f"White Noise (SNR = {match.group(1)} dB)"

    match = re.fullmatch(rf"PitchShiftTest\(n_steps={_NUMBER}\)", test_case_str)
    if match:
        n_str = match.group(1)
        n = float(n_str)
        sign = "+" if n > 0 else ""
        plural = "" if abs(n) == 1 else "s"

        # Strip trailing ".0" for integer values
        n_disp = n_str[:-2] if n_str.endswith(".0") else n_str
        return f"Pitch Shift ({sign}{n_disp} Semitone{plural})"

    match = re.fullmatch(rf"DistortionTest\(drive_db={_NUMBER}\)", test_case_str)
    if match:
        return f"Distortion (Drive = {match.group(1)} dB)"

    match = re.fullmatch(rf"ClippingDbTest\(threshold_db={_NUMBER}\)", test_case_str)
    if match:
        return f"Clipping (Threshold = {match.group(1)} dB)"

    if test_case_str.startswith("ReverbTest"):
        return "Reverb"

    # Unknown -> escape underscores so LaTeX doesn't choke, but otherwise
    # leave it alone so it's obvious in the rendered output.
    return _latex_escape(test_case_str)


def generate_latex_table(
        results_df: pd.DataFrame,
        *,
        backends: tuple[str, str] = ("PCA", "Shazam"),
        test_case_order: list[str] | None = None,
        test_case_labels: dict[str, str] | None = None,
        metrics: Iterable[Metric] | None = None,
        metric_headers: dict[str, str] | None = None,
        metric_format: dict[str, str] | None = None,
        label: str = "tab:benchmark_results",
        caption: str | None = None,
        row_spacing_pt: int = 4,
        include_preamble: bool = False,
        missing_marker: str = "--",
) -> str:
    """
    Generate a LaTeX table comparing two backends with split-cell formatting.

    :param results_df: Output of ``BenchmarkRunner.get_results_df()``. Must contain ``Backend`` and ``Test Case`` columns.
    :param backends: Exactly two backend names. The first goes top-left, the second goes bottom-right (matching the colour-coded splitcell macro).
    :param test_case_order: Explicit row order. If ``None``, uses the natural sorted order of test cases found in ``results_df``.
    :param test_case_labels: Map of ``str(test_case) -> display label``.
        Missing entries fall back to a heuristic that handles the standard
        test cases (CleanTrackTest, WhiteNoiseTest, PitchShiftTest,
        DistortionTest, ClippingDbTest, ReverbTest).
    :param metrics: Metric instances to use as columns. Defaults to Recall, FPR, Precision, and Mean Query Time.
    :param metric_headers: Override for column header LaTeX per metric name.
    :param metric_format: Override for value formatting per metric name.
    :param label: LaTeX ``\\label`` value.
    :param caption: LaTeX caption. If ``None``, a sensible default is built.
    :param row_spacing_pt: Extra vertical space between rows in points.
    :param include_preamble: Prepend the colour + macro definitions.
    :param missing_marker: String used when a metric returns ``None`` for a cell (e.g. no positive queries for that test case).
    """
    if results_df.empty:
        raise ValueError("results_df is empty - run the experiment first.")

    if len(backends) != 2:
        raise ValueError(
            f"This table format uses splitcell which expects exactly 2 "
            f"backends, got {len(backends)}: {backends!r}."
        )

    actual_backends = set(results_df["Backend"].unique())
    missing_backends = [b for b in backends if b not in actual_backends]
    if missing_backends:
        raise ValueError(
            f"Backends not found in results_df: {missing_backends}. "
            f"Available: {sorted(actual_backends)}."
        )

    metrics = list(metrics) if metrics is not None else list(DEFAULT_TABLE_METRICS)
    headers = {**DEFAULT_METRIC_HEADERS, **(metric_headers or {})}
    fmts = {**DEFAULT_METRIC_FORMAT, **(metric_format or {})}

    # Compute everything up front
    cell_values = _compute_per_cell_values(results_df, metrics)

    all_test_cases = list(results_df["Test Case"].unique())
    if test_case_order is None:
        test_case_order = sorted(all_test_cases)
    else:
        unknown = [tc for tc in test_case_order if tc not in all_test_cases]
        if unknown:
            raise ValueError(
                f"test_case_order contains entries not in results_df: {unknown}. "
                f"Available: {sorted(all_test_cases)}."
            )

    # Resolve display labels
    user_labels = test_case_labels or {}
    resolved_labels = {
        tc: user_labels.get(tc, _default_pretty_name(tc))
        for tc in test_case_order
    }

    backend_a, backend_b = backends

    if caption is None:
        caption = (
            f"Performance of the {_latex_escape(backend_a)}-based system and "
            f"{_latex_escape(backend_b)}-style system under different audio "
            f"transformations. Each cell shows "
            f"\\colorbox{{pcacolor!70}}{{\\scriptsize {_latex_escape(backend_a)}}} "
            f"(top-left) vs "
            f"\\colorbox{{shazamcolor!70}}{{\\scriptsize {_latex_escape(backend_b)}}} "
            f"(bottom-right)."
        )

    # Build the table
    col_spec = "@{}l" + "c" * len(metrics) + "@{}"
    header_cells = " & ".join(
        headers.get(m.name, f"\\textbf{{{m.label}}}") for m in metrics
    )

    lines: list[str] = []
    if include_preamble:
        lines.append(PREAMBLE)
        lines.append("")

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{{caption}}}")
    lines.append(rf"    \label{{{label}}}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(rf"    \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")
    lines.append(rf"        \textbf{{Test Case}} & {header_cells} \\")
    lines.append(r"        \midrule")

    last = len(test_case_order) - 1
    spacing = rf"\\[{row_spacing_pt}pt]"

    for i, tc in enumerate(test_case_order):
        cells = []
        for m in metrics:
            fmt = fmts.get(m.name, ".2f")
            val_a = cell_values.get((backend_a, tc), {}).get(m.name)
            val_b = cell_values.get((backend_b, tc), {}).get(m.name)
            cells.append(
                rf"\splitcell{{{_format_cell(val_a, fmt, missing_marker)}}}"
                rf"{{{_format_cell(val_b, fmt, missing_marker)}}}"
            )

        ending = spacing if i < last else r"\\"
        lines.append(
            f"        {resolved_labels[tc]} & " + " & ".join(cells) + f" {ending}"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    }")
    lines.append(r"\end{table}")

    return "\n".join(lines)
