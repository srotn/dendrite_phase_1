from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DENDRITE = REPO / "experiments" / "dendrite"
TASKS = [
    ("01_forced", "01_forced_oscillation", 50_000, 10_000, 20261016),
    ("02_thermal", "03_thermal_field", 30_000, 6_000, 20261017),
    ("03_optical", "04_optical_stack", 30_000, 6_000, 20261017),
    ("04_impedance", "05_impedance_network", 30_000, 6_000, 20261017),
]
MODULE_ROOT = "experiments.dendrite.phase_1_synthetic_experiments"


def load_generator(folder: str):
    return importlib.import_module(f"{MODULE_ROOT}.{folder}.experiment")


def standardize_targets(train: dict, *other_splits: dict) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train["y"].mean(0)
    scale = train["y"].std(0).clamp_min(1e-6)
    train["y"] = (train["y"] - mean) / scale
    for split in other_splits:
        split["y"] = (split["y"] - mean) / scale
    return mean, scale


def generate(task: str, folder: str, train_size: int, valid_size: int, seed: int, smoke: bool = False) -> None:
    module = load_generator(folder)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_size = 16 if smoke else train_size
    valid_size = 8 if smoke else valid_size

    if task == "01_forced":
        train = module.generate(train_size, seed)
        valid = module.generate(valid_size, seed + 1)
        mean, scale = standardize_targets(train, valid)
        dataset = {"train": train, "valid": valid, "y_mean": mean, "y_scale": scale}
    elif task == "04_impedance":
        train = module.generate(train_size, seed, device="cpu", ood=False)
        iid = module.generate(valid_size, seed + 1, device="cpu", ood=False)
        ood = module.generate(valid_size, seed + 2, device="cpu", ood=True)
        mean, scale = standardize_targets(train, iid, ood)
        dataset = {"train": train, "iid": iid, "ood": ood, "y_mean": mean, "y_scale": scale}
    else:
        train = module.generate(train_size, seed, device="cpu")
        valid = module.generate(valid_size, seed + 1, device="cpu")
        mean, scale = standardize_targets(train, valid)
        dataset = {"train": train, "valid": valid, "y_mean": mean, "y_scale": scale}

    for split in (v for v in dataset.values() if isinstance(v, dict)):
        for key, value in split.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite values in {task}/{key}")

    if smoke:
        print(f"PASS {task}: generated {len(train['y'])} training rows")
        return

    out = DENDRITE / "phase_1_synthetic_experiments" / folder / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out / "dataset.pt")
    manifest = {
        "task": task,
        "generator": f"{MODULE_ROOT}.{folder}.experiment",
        "seed": seed,
        "train_rows": len(train["y"]),
        "validation_rows": valid_size,
        "target_mean": mean.tolist(),
        "target_scale": scale.tolist(),
        "ood_rows": len(dataset["ood"]["y"]) if "ood" in dataset else 0,
    }
    (out / "dataset_manifest_public.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {task}: {out / 'dataset.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Generate small samples without writing data")
    args = parser.parse_args()
    for task, folder, train_size, valid_size, seed in TASKS:
        generate(task, folder, train_size, valid_size, seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
