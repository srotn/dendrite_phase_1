"""Concrete Compressive Strength phase-1 training test.

This task is intentionally isolated from the synthetic dendrite experiments.
The baseline follows the paper's Concrete ``DD(4+1)`` order: four multiplicative
DD modules followed by one linear output module. A constant x0 coordinate is
included inside every model, matching the original DD MATLAB formulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from plot_results import plot_results


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parents[2] / "paper_regression_datasets" / "data" / "09_concrete_compressive_strength" / "Concrete_Data.xls"
ARTIFACTS = ROOT / "artifacts"
SEED = 20261017
TRAIN_FRACTION = 0.80
RECIPROCAL_QUANTILE = 0.995
EPSILON = 1e-6


def load_concrete() -> torch.Tensor:
    frame = pd.read_excel(DATA_PATH, engine="xlrd")
    data = torch.tensor(frame.to_numpy(dtype=np.float32), dtype=torch.float32)
    if data.shape != (1030, 9):
        raise ValueError(f"Unexpected Concrete shape: {tuple(data.shape)}")
    if not torch.isfinite(data).all():
        raise ValueError("Concrete data contains non-finite values")
    return data


def minmax_fit(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower = values.min(dim=0).values
    upper = values.max(dim=0).values
    scale = (upper - lower).clamp_min(EPSILON)
    return lower, scale


def minmax_transform(values: torch.Tensor, lower: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return 2.0 * (values - lower) / scale - 1.0


def reciprocal(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs().clamp_min(EPSILON)
    result = values.sign() / magnitude
    return torch.where(values == 0, torch.zeros_like(result), result)


def prepare_dataset(seed: int) -> dict:
    data = load_concrete()
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(data), generator=generator)
    train_size = int(round(len(data) * TRAIN_FRACTION))
    train_indices = permutation[:train_size]
    valid_indices = permutation[train_size:]
    physical_train = data[train_indices, :8]
    physical_valid = data[valid_indices, :8]
    target_train_raw = data[train_indices, 8:]
    target_valid_raw = data[valid_indices, 8:]

    input_lower, input_scale = minmax_fit(physical_train)
    target_lower, target_scale = minmax_fit(target_train_raw)
    x_train = minmax_transform(physical_train, input_lower, input_scale)
    x_valid = minmax_transform(physical_valid, input_lower, input_scale)
    y_train = minmax_transform(target_train_raw, target_lower, target_scale)
    y_valid = minmax_transform(target_valid_raw, target_lower, target_scale)

    reciprocal_train_raw = reciprocal(physical_train)
    reciprocal_valid_raw = reciprocal(physical_valid)
    reciprocal_cap = torch.quantile(reciprocal_train_raw.abs(), RECIPROCAL_QUANTILE, dim=0).clamp_min(EPSILON)
    reciprocal_train_capped = torch.sign(reciprocal_train_raw) * torch.minimum(reciprocal_train_raw.abs(), reciprocal_cap)
    reciprocal_valid_capped = torch.sign(reciprocal_valid_raw) * torch.minimum(reciprocal_valid_raw.abs(), reciprocal_cap)
    reciprocal_lower, reciprocal_scale = minmax_fit(reciprocal_train_capped)
    reciprocal_train = minmax_transform(reciprocal_train_capped, reciprocal_lower, reciprocal_scale)
    reciprocal_valid = minmax_transform(reciprocal_valid_capped, reciprocal_lower, reciprocal_scale)

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_train_raw": target_train_raw,
        "y_valid_raw": target_valid_raw,
        "reciprocal_train": reciprocal_train,
        "reciprocal_valid": reciprocal_valid,
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "input_lower": input_lower,
        "input_scale": input_scale,
        "target_lower": target_lower,
        "target_scale": target_scale,
        "reciprocal_cap": reciprocal_cap,
        "reciprocal_lower": reciprocal_lower,
        "reciprocal_scale": reciprocal_scale,
    }


class PaperDD(nn.Module):
    def __init__(self, input_width: int, layers: int, output_dim: int = 1):
        super().__init__()
        self.input_width = input_width
        self.layers = nn.ModuleList(
            nn.Linear(input_width + 1, input_width + 1, bias=False)
            for _ in range(layers)
        )
        self.output = nn.Linear(input_width + 1, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = torch.cat((torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x), dim=1)
        hidden = anchor
        for layer in self.layers:
            hidden = layer(hidden) * anchor
        return self.output(hidden)


class BaseDD(nn.Module):
    name = "base_dd"

    def __init__(self):
        super().__init__()
        self.dd = PaperDD(input_width=8, layers=4)

    def forward(self, x: torch.Tensor, reciprocal_features: torch.Tensor | None = None) -> torch.Tensor:
        del reciprocal_features
        return self.dd(x)


class DuplicateConcatDD(nn.Module):
    name = "duplicate_concat_dd"

    def __init__(self):
        super().__init__()
        self.dd = PaperDD(input_width=16, layers=4)

    def forward(self, x: torch.Tensor, reciprocal_features: torch.Tensor | None = None) -> torch.Tensor:
        del reciprocal_features
        return self.dd(torch.cat((x, x), dim=1))


class DeepBaseDD(nn.Module):
    name = "deep_base_dd_12_layers"

    def __init__(self):
        super().__init__()
        self.dd = PaperDD(input_width=8, layers=16)

    def forward(self, x: torch.Tensor, reciprocal_features: torch.Tensor | None = None) -> torch.Tensor:
        del reciprocal_features
        return self.dd(x)


class ReciprocalConcatDD(nn.Module):
    name = "reciprocal_concat_dd"

    def __init__(self):
        super().__init__()
        self.dd = PaperDD(input_width=16, layers=4)

    def forward(self, x: torch.Tensor, reciprocal_features: torch.Tensor | None = None) -> torch.Tensor:
        if reciprocal_features is None:
            raise ValueError("reciprocal features are required")
        return self.dd(torch.cat((x, reciprocal_features), dim=1))


class SafePhysicalReciprocalConcatDD(ReciprocalConcatDD):
    name = "safe_physical_reciprocal_concat_dd"


MODELS = (BaseDD, DuplicateConcatDD, DeepBaseDD, ReciprocalConcatDD, SafePhysicalReciprocalConcatDD)


def parameter_counts() -> dict[str, int]:
    return {model.name: sum(parameter.numel() for parameter in model().parameters()) for model in MODELS}


def inverse_target(values: torch.Tensor, lower: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (values + 1.0) * scale / 2.0 + lower


def metrics(prediction: torch.Tensor, target: torch.Tensor, lower: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    actual = inverse_target(prediction, lower, scale)
    truth = inverse_target(target, lower, scale)
    error = actual - truth
    return {
        "mse_normalized": float((prediction - target).square().mean()),
        "mse_raw": float(error.square().mean()),
        "rmse_raw": float(error.square().mean().sqrt()),
        "mae_raw": float(error.abs().mean()),
    }


def train_model(model_cls, dataset: dict, args: argparse.Namespace, device: torch.device) -> dict:
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(dataset["x_train"], dataset["reciprocal_train"], dataset["y_train"]),
        batch_size=args.batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        TensorDataset(dataset["x_valid"], dataset["reciprocal_valid"], dataset["y_valid"]),
        batch_size=args.batch,
    )
    folder = ARTIFACTS / model_cls.name
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "metrics.jsonl"
    checkpoint_path = folder / "best.pt"
    best = math.inf
    best_epoch = 0
    best_row = None
    start_epoch = 1
    existing_rows = []
    if args.resume and log_path.is_file() and checkpoint_path.is_file():
        existing_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if existing_rows:
            start_epoch = int(existing_rows[-1]["epoch"]) + 1
            best_row = min(existing_rows, key=lambda row: float(row["mse_normalized"]))
            best = float(best_row["mse_normalized"])
            best_epoch = int(best_row["epoch"])
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
    else:
        log_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    started_total = time.perf_counter()
    mode = "a" if existing_rows else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            losses = []
            gradient_norms = []
            for x, reciprocal_features, target in loader:
                x = x.to(device)
                reciprocal_features = reciprocal_features.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, reciprocal_features)
                loss = (prediction - target).square().mean()
                loss.backward()
                gradient_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
                optimizer.step()
                losses.append(float(loss.detach()))

            model.eval()
            predictions = []
            targets = []
            with torch.inference_mode():
                for x, reciprocal_features, target in valid_loader:
                    predictions.append(model(x.to(device), reciprocal_features.to(device)).cpu())
                    targets.append(target)
            prediction = torch.cat(predictions)
            target = torch.cat(targets)
            row = {
                "epoch": epoch,
                "seed": args.seed,
                "model": model_cls.name,
                "seconds": time.perf_counter() - started,
                "train_loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(gradient_norms)),
                **metrics(prediction, target, dataset["target_lower"], dataset["target_scale"]),
            }
            log.write(json.dumps(row) + "\n")
            if row["mse_normalized"] < best:
                best = row["mse_normalized"]
                best_epoch = epoch
                best_row = row
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": row,
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    },
                    checkpoint_path,
                )
            if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
                print(json.dumps(row), flush=True)
    return {
        "model": model_cls.name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_mse_normalized": best,
        "best_metrics": best_row,
        "seconds": time.perf_counter() - started_total,
        "metrics": str(log_path.relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "resume": bool(existing_rows),
        "resume_start_epoch": start_epoch if existing_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    dataset = prepare_dataset(args.seed)
    torch.save(dataset, ARTIFACTS / "dataset.pt")
    counts = parameter_counts()
    manifest = {
        "task": "Concrete Compressive Strength",
        "experiment_group": "paper_regression_reproduction/09_concrete_compressive_strength/phase1_training_test",
        "protocol": "one_seed_2000_epochs_training_test",
        "paper_protocol_mode": "training_test_only_no_100_repeat_stop_validation",
        "resume": bool(args.resume),
        "resume_source": "5000_epoch_best_checkpoints" if args.resume else None,
        "optimizer_state_resumed": False if args.resume else None,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "learning_rate": args.lr,
        "device": str(device),
        "train_size": int(dataset["x_train"].shape[0]),
        "valid_size": int(dataset["x_valid"].shape[0]),
        "normalization": "train_split_minmax_to_minus_one_one",
        "baseline": "paper_DD(4+1)_with_constant_x0_coordinate",
        "deep_layers": 16,
        "safe_reciprocal": {
            "source": "raw_concrete_features",
            "epsilon": EPSILON,
            "cap_quantile": RECIPROCAL_QUANTILE,
            "normalization": "train_split_minmax_to_minus_one_one",
        },
        "models": [model.name for model in MODELS],
        "parameter_counts": counts,
        "data_source": "paper_regression_datasets/data/09_concrete_compressive_strength/Concrete_Data.xls",
    }
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "parameter_report.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    results = [train_model(model, dataset, args, device) for model in MODELS]
    summary = {"manifest": manifest, "results": results}
    summary["plots"] = plot_results(ARTIFACTS)
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
