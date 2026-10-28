from __future__ import annotations

import hashlib
import importlib
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
RAW = REPO / "data" / "raw" / "uci_165"
DATASET_MODULE = (
    "experiments.dendrite.phase_2_paper_regression.paper_regression_reproduction."
    "09_concrete_compressive_strength.phase1_training_test.experiment"
)
CONCRETE_MODULE_DIR = (
    REPO
    / "experiments"
    / "dendrite"
    / "phase_2_paper_regression"
    / "paper_regression_reproduction"
    / "09_concrete_compressive_strength"
    / "phase1_training_test"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "concrete_compressive_strength.zip"
    output = (
        REPO
        / "experiments"
        / "dendrite"
        / "phase_2_paper_regression"
        / "paper_regression_datasets"
        / "data"
        / "09_concrete_compressive_strength"
        / "Concrete_Data.xls"
    )
    if not archive.is_file():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        with zipfile.ZipFile(temporary) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("The UCI archive failed its CRC check")
            member = next(
                (name for name in zipped.namelist() if PurePosixPath(name).name.lower() == "concrete_data.xls"),
                None,
            )
            if member is None:
                raise FileNotFoundError("Concrete_Data.xls was not found in the UCI archive")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(zipped.read(member))
        temporary.replace(archive)
    elif not output.is_file():
        with zipfile.ZipFile(archive) as zipped:
            member = next(
                (name for name in zipped.namelist() if PurePosixPath(name).name.lower() == "concrete_data.xls"),
                None,
            )
            if member is None:
                raise FileNotFoundError("Concrete_Data.xls was not found in the cached UCI archive")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(zipped.read(member))

    if str(CONCRETE_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(CONCRETE_MODULE_DIR))
    module = importlib.import_module(DATASET_MODULE)
    dataset = module.prepare_dataset(seed=20261017)
    module.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = module.ARTIFACTS / "dataset.pt"
    torch.save(dataset, output)
    print(f"Wrote {output}; archive SHA-256 {sha256(archive)}")


if __name__ == "__main__":
    main()
