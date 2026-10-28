"""Plot per-model training curves for the isolated Yacht experiment."""

from __future__ import annotations

import json
from pathlib import Path


def plot_results(artifacts: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_dirs = sorted(path for path in artifacts.iterdir() if path.is_dir() and (path / "metrics.jsonl").is_file())
    if not model_dirs:
        raise FileNotFoundError(f"No model metrics found under {artifacts}")
    curves = {}
    for model_dir in model_dirs:
        rows = [json.loads(line) for line in (model_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        curves[model_dir.name] = rows

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)
    labels = {
        "train_loss": "Training loss",
        "mse_normalized": "Validation normalized MSE",
        "rmse_raw": "Validation raw RMSE",
    }
    for model, rows in curves.items():
        epochs = [row["epoch"] for row in rows]
        for axis, metric in zip(axes, labels):
            axis.plot(epochs, [row[metric] for row in rows], label=model, linewidth=1.4)
    for axis, metric in zip(axes, labels):
        axis.set_title(labels[metric])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Concrete Compressive Strength: isolated phase-1 training test")
    figure.tight_layout()
    png = artifacts / "training_curves.png"
    pdf = artifacts / "training_curves.pdf"
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [str(png), str(pdf)]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    paths = plot_results(root / "artifacts")
    summary_path = root / "artifacts" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["plots"] = paths
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(paths))
