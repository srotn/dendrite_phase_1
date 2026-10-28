"""Regenerate the paper's seven statistical figures from audited summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


FIG_DIR = Path(__file__).resolve().parent
PAPER_DIR = FIG_DIR.parent
ROOT = PAPER_DIR.parent
STYLE = FIG_DIR / "science_paper.mplstyle"
TASKS = [
    "01_forced",
    "02_thermal",
    "03_optical",
    "04_impedance",
    "05_concrete",
    "06_co",
]
TASK_NAMES = {
    "01_forced": "Forced oscillation",
    "02_thermal": "Thermal field",
    "03_optical": "Optical regression",
    "04_impedance": "Impedance",
    "05_concrete": "Concrete strength",
    "06_co": "CO sensor",
}
COLORS = {
    "duplicate": "#474747",
    "reciprocal": "#0C5DA5",
    "original": "#FF9500",
    "matched": "#00B945",
    "polynomial": "#FF2C00",
    "tree": "#845B97",
    "budget_2000": "#0C5DA5",
    "budget_4000": "#FF9500",
    "budget_8000": "#00B945",
}
DEPTHS_PRIMARY = [1, 2, 3, 4, 8, 16]
DEPTHS_FOLLOWUP = [1, 4, 8, 16]


plt.style.use(str(STYLE))
matplotlib.rcParams.update(
    {
        "figure.figsize": (5.30, 4.30),
        "font.family": "serif",
        "font.size": 9.0,
        "axes.labelsize": 9.8,
        "axes.titlesize": 9.8,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 8.2,
        "figure.titlesize": 10.2,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def small_multiples() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(5.30, 4.42))
    for task, ax in zip(TASKS, axes.flat):
        ax.set_title(TASK_NAMES[task], pad=4)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", pad=2.0, length=2.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, axes


def save_pdf(fig: plt.Figure, name: str) -> None:
    output = FIG_DIR / name
    fig.savefig(
        output,
        format="pdf",
        bbox_inches=None,
        metadata={"Creator": "regenerate_figures.py; SciencePlots science style adaptation"},
    )
    plt.close(fig)
    if not output.is_file() or output.stat().st_size < 4000:
        raise RuntimeError(f"Figure output missing or unexpectedly small: {output}")
    print(f"Wrote {output.name}: {output.stat().st_size:,} bytes")


def finish_small_multiples(
    fig: plt.Figure,
    handles: list[Line2D],
    labels: list[str],
    legend_columns: int = 2,
    top: float = 0.80,
) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=legend_columns,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
        labelspacing=0.35,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.15, top=top, wspace=0.31, hspace=0.43)


def plot_depth_gain() -> None:
    with (ROOT / "01_independent_evaluation/analysis/gain_summary.json").open(encoding="utf-8") as f:
        rows = json.load(f)
    fig, axes = small_multiples()
    controls = [
        ("base", "Original-input control", COLORS["duplicate"], "o", "-"),
        ("duplicate", "Duplicate-input control", COLORS["reciprocal"], "s", "--"),
    ]
    for task, ax in zip(TASKS, axes.flat):
        for control, _, color, marker, linestyle in controls:
            values = sorted(
                (r for r in rows if r["task"] == task and r["control"] == control),
                key=lambda r: r["depth"],
            )
            x = np.array([r["depth"] for r in values], dtype=float)
            mean = np.array([r["mean"] for r in values], dtype=float)
            low = np.array([r["low"] for r in values], dtype=float)
            high = np.array([r["high"] for r in values], dtype=float)
            ax.plot(x, mean, color=color, marker=marker, linestyle=linestyle, zorder=3)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.axhline(0, color="#777777", linewidth=0.75, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEPTHS_PRIMARY, labels=[str(d) for d in DEPTHS_PRIMARY])
        ax.set_xlim(0.88, 18)
    finish_small_multiples(
        fig,
        [
            Line2D([], [], color=COLORS["duplicate"], marker="o", linestyle="-"),
            Line2D([], [], color=COLORS["reciprocal"], marker="s", linestyle="--"),
        ],
        ["Original-input control", "Duplicate-input control"],
    )
    fig.supxlabel("Multiplicative depth", y=0.035, fontsize=9.8)
    fig.supylabel(r"RMSE reduction vs control (\%)", x=0.018, fontsize=9.8)
    save_pdf(fig, "depth_gain_by_task.pdf")


def plot_external_baselines() -> None:
    frame = read_csv(
        "03_optimization_fairness/analysis/fair_comparison_summary.csv"
    )
    frame = frame[frame.domain.isin(["iid", "test"])].copy()
    expected = {
        "dd_duplicate",
        "dd_reciprocal",
        "mlp_base",
        "mlp_reciprocal",
        "polynomial_ridge",
        "hist_gradient_boosting",
    }
    if set(frame.method.unique()) != expected:
        raise ValueError(f"External-comparison methods differ from protocol: {sorted(frame.method.unique())}")

    series = [
        ("dd_duplicate", "DD duplicate", COLORS["duplicate"], "o", "-", True),
        ("dd_reciprocal", "DD reciprocal", COLORS["reciprocal"], "D", "-", True),
        ("mlp_base", "MLP original", COLORS["original"], "o", "--", True),
        ("mlp_reciprocal", "MLP reciprocal", COLORS["matched"], "D", "--", True),
        ("polynomial_ridge", "Polynomial ridge reference", COLORS["polynomial"], "X", "None", False),
        ("hist_gradient_boosting", "Gradient-boosting reference", COLORS["tree"], "P", "None", False),
    ]
    fig, axes = small_multiples()
    for task, ax in zip(TASKS, axes.flat):
        part = frame[frame.task == task]
        for method, _, color, marker, linestyle, has_depth_curve in series:
            values = part[part.method == method].groupby("depth", as_index=False).mean(numeric_only=True)
            values = values.sort_values("depth")
            if has_depth_curve:
                if list(values.depth.astype(int)) != [4, 16]:
                    raise ValueError(f"Expected depth-4/16 points for {method} on {task}")
                ax.plot([0, 1], values["mean"].to_numpy(), color=color, marker=marker, linestyle=linestyle, zorder=3)
            else:
                if list(values.depth.astype(int)) != [4]:
                    raise ValueError(f"Expected one classical reference value for {method} on {task}")
                ax.plot([2], values["mean"], color=color, marker=marker, linestyle="None", zorder=4)
        ax.axvline(1.5, color="#A0A0A0", linewidth=0.7, linestyle=(0, (2, 2)))
        ax.set_xticks([0, 1, 2], labels=["d = 4", "d = 16", "Classical"])
        ax.set_xlim(-0.32, 2.56)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    handles = [
        Line2D([], [], color=color, marker=marker, linestyle=linestyle)
        for _, _, color, marker, linestyle, _ in series
    ]
    labels = [entry[1] for entry in series]
    finish_small_multiples(fig, handles, labels, legend_columns=3, top=0.75)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.18, top=0.75, wspace=0.31, hspace=0.43)
    fig.supxlabel("Neural-model depth; separate reference category for classical models", y=0.035, fontsize=9.4)
    fig.supylabel("Normalized test RMSE", x=0.018, fontsize=9.8)
    save_pdf(fig, "external_baseline_comparison.pdf")


def plot_budget_gains() -> None:
    frame = read_csv("03_optimization_fairness/analysis/paired_budget_gains.csv")
    frame = frame[frame.domain.isin(["iid", "test"])].copy()
    fig, axes = small_multiples()
    budgets = [2000, 4000, 8000]
    for task, ax in zip(TASKS, axes.flat):
        part = frame[frame.task == task]
        for budget, color in zip(budgets, [COLORS["budget_2000"], COLORS["budget_4000"], COLORS["budget_8000"]]):
            z = part[part.budget == budget]
            z = z.groupby(["fold", "depth"], as_index=False).gain_percent.mean()
            z = z.groupby("depth", as_index=False).gain_percent.mean().sort_values("depth")
            ax.plot(z.depth, z.gain_percent, marker="o", color=color, label=f"{budget:,}")
        ax.axhline(0, color="#777777", linewidth=0.75, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEPTHS_FOLLOWUP, labels=[str(d) for d in DEPTHS_FOLLOWUP])
        ax.set_xlim(0.82, 19)
    handles = [
        Line2D([], [], color=color, marker="o", linestyle="-", label=f"{budget:,} epochs")
        for budget, color in zip(budgets, [COLORS["budget_2000"], COLORS["budget_4000"], COLORS["budget_8000"]])
    ]
    finish_small_multiples(fig, handles, [h.get_label() for h in handles], legend_columns=3)
    fig.supxlabel("Multiplicative depth", y=0.035, fontsize=9.8)
    fig.supylabel(r"RMSE gain vs duplicate input (\%)", x=0.018, fontsize=9.8)
    save_pdf(fig, "budget_gain_curves.pdf")


def plot_fixed_lr_budget() -> None:
    frame = read_csv("03_optimization_fairness/analysis/fixed_lr_budget_summary.csv")
    frame = frame[(frame.depth == 16) & frame.domain.isin(["iid", "test"])].copy()
    frame["lr_label"] = frame.lr.map(lambda v: f"{float(v):g}")
    lrs = sorted(frame.lr.unique())
    colors = [COLORS["reciprocal"], COLORS["original"], COLORS["matched"]]
    fig, axes = small_multiples()
    for task, ax in zip(TASKS, axes.flat):
        part = frame[frame.task == task]
        for lr, color in zip(lrs, colors):
            z = part[part.lr == lr].groupby("budget", as_index=False).gain_percent.mean().sort_values("budget")
            ax.plot(z.budget / 1000, z.gain_percent, color=color, marker="o", label=f"{float(lr):g}")
        ax.axhline(0, color="#777777", linewidth=0.75, zorder=1)
        ax.set_xticks([2, 4, 8], labels=["2k", "4k", "8k"])
        ax.set_xlim(1.5, 8.5)
    handles = []
    for lr, color in zip(lrs, colors):
        exponent = int(round(np.log10(float(lr))))
        mantissa = float(lr) / (10**exponent)
        label = rf"$\eta$ = {mantissa:g}e{exponent}"
        handles.append(Line2D([], [], color=color, marker="o", linestyle="-", label=label))
    finish_small_multiples(fig, handles, [h.get_label() for h in handles], legend_columns=3)
    fig.supxlabel("Continuous training budget (epochs)", y=0.035, fontsize=9.8)
    fig.supylabel(r"RMSE gain vs duplicate input (\%)", x=0.018, fontsize=9.8)
    save_pdf(fig, "fixed_learning_rate_budget.pdf")


def mechanism_means(frame: pd.DataFrame, task: str, key: str) -> pd.Series:
    part = frame[(frame.task == task) & frame.domain.isin(["iid", "test"])]
    return part.groupby(["fold", "depth"])[key].mean().groupby("depth").mean().sort_index()


def plot_mechanism_controls() -> None:
    frame = read_csv("04_mechanism_controls/analysis/paired_mechanism_effects.csv")
    series = [
        ("gain_c0_reciprocal", "Native, slot 0", COLORS["duplicate"], "o", "-"),
        ("gain_c1_reciprocal", "Native, slot 1", COLORS["reciprocal"], "s", "-"),
        ("gain_c0_matched", "Moment-matched, slot 0", COLORS["original"], "o", "--"),
        ("gain_c1_matched", "Moment-matched, slot 1", COLORS["matched"], "s", "--"),
    ]
    fig, axes = small_multiples()
    for task, ax in zip(TASKS, axes.flat):
        for key, _, color, marker, linestyle in series:
            z = mechanism_means(frame, task, key)
            ax.plot(z.index, z.values, color=color, marker=marker, linestyle=linestyle)
        ax.axhline(0, color="#777777", linewidth=0.75, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], labels=["1", "4", "16"])
        ax.set_xlim(0.78, 20)
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls) for _, _, c, m, ls in series]
    finish_small_multiples(fig, handles, [s[1] for s in series], legend_columns=2)
    fig.supxlabel("Multiplicative depth", y=0.035, fontsize=9.8)
    fig.supylabel(r"RMSE gain vs matched duplicate (\%)", x=0.018, fontsize=9.8)
    save_pdf(fig, "mechanism_constant_moments.pdf")


def plot_field_subsets() -> None:
    frame = read_csv("04_mechanism_controls/analysis/paired_mechanism_effects.csv")
    series = [
        ("gain_c1_matched", "Full reciprocal representation", COLORS["duplicate"], "o", "-"),
        ("gain_c1_subset", "Prespecified field subset", COLORS["reciprocal"], "s", "-"),
        ("gain_c1_complement", "Complement field subset", COLORS["original"], "D", "--"),
    ]
    fig, axes = small_multiples()
    for task, ax in zip(TASKS, axes.flat):
        for key, _, color, marker, linestyle in series:
            z = mechanism_means(frame, task, key)
            ax.plot(z.index, z.values, color=color, marker=marker, linestyle=linestyle)
        ax.axhline(0, color="#777777", linewidth=0.75, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], labels=["1", "4", "16"])
        ax.set_xlim(0.78, 20)
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls) for _, _, c, m, ls in series]
    finish_small_multiples(fig, handles, [s[1] for s in series], legend_columns=3)
    fig.supxlabel("Multiplicative depth", y=0.035, fontsize=9.8)
    fig.supylabel(r"RMSE gain vs matched duplicate (\%)", x=0.018, fontsize=9.8)
    save_pdf(fig, "mechanism_field_subsets.pdf")


def plot_impedance_ood() -> None:
    frame = read_csv("analysis/final_impedance_ood_absolute_log_voltage.csv")
    names = {
        "dd_duplicate": "Duplicate input",
        "dd_reciprocal": "Reciprocal input",
    }
    colors = {"dd_duplicate": COLORS["duplicate"], "dd_reciprocal": COLORS["reciprocal"]}
    markers = {"dd_duplicate": "o", "dd_reciprocal": "s"}
    fig, ax = plt.subplots(figsize=(5.30, 3.45))
    for method in ["dd_duplicate", "dd_reciprocal"]:
        z = frame[frame.method == method].sort_values("depth")
        ax.plot(z.depth, z["mean"], color=colors[method], marker=markers[method], label=names[method])
        ax.fill_between(z.depth.to_numpy(), z["min"].to_numpy(), z["max"].to_numpy(), color=colors[method], alpha=0.14, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 8, 16], labels=["1", "4", "8", "16"])
    ax.set_xlim(0.82, 19)
    ax.set_xlabel("DD multiplicative depth")
    ax.set_ylabel("Absolute OOD RMSE (log-voltage units)")
    ax.grid(axis="both")
    ax.legend(loc="upper left", ncol=2)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.18, top=0.98)
    save_pdf(fig, "impedance_ood_absolute_error.pdf")


def main() -> None:
    plot_depth_gain()
    plot_external_baselines()
    plot_budget_gains()
    plot_fixed_lr_budget()
    plot_mechanism_controls()
    plot_field_subsets()
    plot_impedance_ood()


if __name__ == "__main__":
    main()
