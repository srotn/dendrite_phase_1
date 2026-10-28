"""Acquire UCI 487, audit complete files and prepare the fixed batch split."""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
ARTIFACTS = ROOT / "artifacts"
URL = "https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip"


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire():
    RAW.mkdir(exist_ok=True)
    archive = RAW / "uci_487.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".part")
        last, total = time.monotonic(), 0
        with urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                total += len(block)
                if time.monotonic() - last > 10:
                    print(json.dumps({"downloaded_MiB": round(total / 2**20, 1)}), flush=True)
                    last = time.monotonic()
        with zipfile.ZipFile(temporary) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Archive CRC check failed")
        temporary.rename(archive)
    target = RAW / "extracted"
    target.mkdir(exist_ok=True)
    for source in [archive]:
        with zipfile.ZipFile(source) as zipped:
            for member in zipped.infolist():
                destination = (target / member.filename).resolve()
                destination.relative_to(target.resolve())
            zipped.extractall(target)
    for source in sorted(target.rglob("*.zip")):
        inner = source.with_suffix("")
        inner.mkdir(exist_ok=True)
        with zipfile.ZipFile(source) as zipped:
            for member in zipped.infolist():
                (inner / member.filename).resolve().relative_to(inner.resolve())
            zipped.extractall(inner)
    paths = sorted(path for path in target.rglob("*") if path.suffix.lower() in {".csv", ".txt"} and path.stem[:8].isdigit())
    if len(paths) != 13:
        raise ValueError(f"Expected 13 acquisition files: {paths}")
    return archive, paths


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    archive, paths = acquire()
    records, pieces = [], {"train": [], "valid": [], "test": []}
    for index, path in enumerate(paths):
        split = "train" if index < 9 else "valid" if index < 11 else "test"
        frame = pd.read_csv(path)
        if frame.shape[1] != 20:
            raise ValueError((path, frame.shape, frame.columns.tolist()))
        values = frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"NaN/Inf in {path.name}")
        original_times = values[:, 0].copy()
        raw_differences = np.diff(original_times)
        source_order = np.argsort(original_times, kind="stable")
        values = values[source_order]
        times = values[:, 0]
        if np.any(values[:, 6:] <= 0):
            raise ValueError(f"Nonpositive physical resistance in {path.name}")
        bins = np.floor((times - times[0]) / 10).astype(np.int64)
        indices = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
        sampled = values[indices]
        record = {
            "file": path.name, "relative_file": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path), "split": split, "rows": len(values), "sampled_rows": len(indices),
            "columns": frame.columns.tolist(), "time_start": times[0], "time_end": times[-1],
            "median_sample_interval_seconds": float(np.median(np.diff(times))),
            "raw_backward_timestamp_steps": int((raw_differences < 0).sum()),
            "raw_equal_timestamp_steps": int((raw_differences == 0).sum()),
            "timestamp_order_rule": "stable ascending timestamp within original acquisition file",
            "duplicate_rows": int(frame.duplicated().sum()), "nonfinite_values": 0,
            "nonpositive_resistance": 0, "co_range_ppm": [float(values[:, 1].min()), float(values[:, 1].max())],
            "co_unique_values": int(len(np.unique(values[:, 1]))),
            "resistance_min_MOhm": values[:, 6:].min(axis=0).tolist(),
            "resistance_max_MOhm": values[:, 6:].max(axis=0).tolist(),
        }
        records.append(record)
        pieces[split].append({"file": path.name, "values": sampled.astype(np.float32), "source_row": source_order[indices]})
        print(json.dumps({key: record[key] for key in ("file", "split", "rows", "sampled_rows", "co_range_ppm")}), flush=True)
    splits = {}
    for split, parts in pieces.items():
        data = np.concatenate([part["values"] for part in parts])
        batch_id = np.concatenate([np.full(len(part["values"]), i, dtype=np.int64) for i, part in enumerate(parts)])
        splits[split] = {
            "physical_x": torch.from_numpy(data[:, 2:].copy()), "raw_y": torch.from_numpy(data[:, 1:2].copy()),
            "batch_id": torch.from_numpy(batch_id), "files": [part["file"] for part in parts],
            "source_rows": [torch.from_numpy(part["source_row"]) for part in parts],
        }
    train_x = splits["train"]["physical_x"]
    feature_mean, feature_std = train_x.mean(0), train_x.std(0, correction=0)
    raw_y = splits["train"]["raw_y"]
    y_mean, y_std = raw_y.mean(0), raw_y.std(0, correction=0)
    safe_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)
    normalized_resistance = (train_x[:, 4:] - feature_mean[4:]) / safe_std[4:]
    normalized_epsilon = 0.05
    normalized_inverse = torch.where(normalized_resistance < 0, -1., 1.) / normalized_resistance.abs().clamp_min(normalized_epsilon)
    physical_epsilon = 1e-6
    conductance = train_x[:, 4:].clamp_min(physical_epsilon).reciprocal()
    ordinary_caps = torch.quantile(normalized_inverse.abs(), .995, dim=0)
    physical_caps = torch.quantile(conductance, .995, dim=0)
    ordinary_train = normalized_inverse.clamp(-ordinary_caps, ordinary_caps)
    physical_train = torch.minimum(conductance, physical_caps)
    ordinary_mean, ordinary_std = ordinary_train.mean(0), ordinary_train.std(0, correction=0).clamp_min(1e-6)
    physical_mean, physical_std = physical_train.mean(0), physical_train.std(0, correction=0).clamp_min(1e-6)
    # Bound all z-scored branches uniformly before the polynomial DD stack.
    # This fixed threshold uses no validation/test targets or performance.
    feature_clip = 3.0
    clipping = {}
    for split_name, split in splits.items():
        x = (split["physical_x"] - feature_mean) / safe_std
        ordinary = torch.where(x[:, 4:] < 0, -1., 1.) / x[:, 4:].abs().clamp_min(normalized_epsilon)
        ordinary = ordinary.clamp(-ordinary_caps, ordinary_caps)
        physical = torch.minimum(split["physical_x"][:, 4:].clamp_min(physical_epsilon).reciprocal(), physical_caps)
        branches = {"x": x, "ordinary": (ordinary - ordinary_mean) / ordinary_std,
                    "physical": (physical - physical_mean) / physical_std}
        clipping[split_name] = {}
        for branch, values in branches.items():
            clipping[split_name][branch] = {
                "values_clipped_per_feature": (values.abs() > feature_clip).sum(0).tolist(),
                "rows_with_any_clipped_value": int((values.abs() > feature_clip).any(1).sum()),
                "unclipped_max_abs": float(values.abs().max()),
            }
            split[branch] = values.clamp(-feature_clip, feature_clip)
        split["y"] = (split["raw_y"] - y_mean) / y_std
    assert not torch.allclose(splits["train"]["ordinary"], splits["train"]["physical"])
    transforms = {"x_mean": feature_mean, "x_std": safe_std, "y_mean": y_mean, "y_std": y_std,
                  "ordinary_caps": ordinary_caps, "physical_caps": physical_caps,
                  "ordinary_mean": ordinary_mean, "ordinary_std": ordinary_std,
                  "physical_mean": physical_mean, "physical_std": physical_std}
    payload = {"splits": splits, "transforms": transforms, "feature_clip": feature_clip}
    torch.save(payload, ARTIFACTS / "dataset.pt")
    report = {
        "source": URL, "archive_sha256": sha256(archive), "files": records,
        "full_rows": sum(record["rows"] for record in records),
        "sampled_rows": {name: len(split["x"]) for name, split in splits.items()},
        "input_names": records[0]["columns"][2:], "target": records[0]["columns"][1],
        "sampling": "first actual observation in each 10-second bin, anchored at file start",
        "split": "chronological whole acquisition files: first 9 train, next 2 valid, last 2 test",
        "feature_std": feature_std.tolist(), "constant_features": [records[0]["columns"][i+2] for i, value in enumerate(feature_std) if value < 1e-6],
        "normalization": "train-only z-score, constant columns centered with scale=1; all input branches clipped to [-3,3]",
        "feature_clip": feature_clip, "clipping": clipping,
        "ordinary_denominator_floor_standardized": normalized_epsilon,
        "physical_denominator_floor_MOhm": physical_epsilon, "reciprocal_cap_quantile": .995,
        "transform_statistics": {key: value.tolist() for key, value in transforms.items()},
        "ordinary_physical_features_distinct": True,
        "train_max_abs": {key: float(splits["train"][key].abs().max()) for key in ["x", "ordinary", "physical"]},
    }
    save_json(ARTIFACTS / "data_audit.json", report)
    print(json.dumps({"prepared": report["sampled_rows"], "full_rows": report["full_rows"], "train_max_abs": report["train_max_abs"]}), flush=True)


if __name__ == "__main__":
    main()
