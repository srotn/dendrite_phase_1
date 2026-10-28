# Data Sources and Preparation

The repository does not redistribute raw or prepared datasets. Downloaded data, generated tensors, checkpoints, and logs are ignored by Git.

## Synthetic Tasks

Four synthetic generators are included under `experiments/dendrite/phase_1_synthetic_experiments/`. `scripts/prepare_synthetic_inputs.py` reproduces the original development inputs and writes them to the paths expected by the Phase 4 and Phase 6 scripts:

| Task | Generator | Training rows | Validation rows | Seed |
|---|---|---:|---:|---:|
| Forced oscillation | `01_forced_oscillation` | 50,000 | 10,000 | 20261016 |
| Thermal field | `03_thermal_field` | 30,000 | 6,000 | 20261017 |
| Optical regression | `04_optical_stack` | 30,000 | 6,000 | 20261017 |
| Impedance | `05_impedance_network` | 30,000 | 6,000 IID, 6,000 OOD | 20261017 |

The script creates derived files under `artifacts/`; these are generated locally and are not committed.

## Concrete Strength

- UCI Machine Learning Repository dataset 165: [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165)
- Official archive: `https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip`
- `scripts/prepare_concrete_inputs.py` downloads the archive, extracts `Concrete_Data.xls` into the path expected by the Phase 2 preprocessing code, and writes the prepared tensor locally.
- Check the UCI page for the dataset's current citation and terms of use.

## CO Sensor Regression

- UCI Machine Learning Repository dataset 487: [Gas Sensor Array Temperature Modulation](https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation)
- DOI: `10.24432/C5S302`
- The included `prepare_data.py` downloads and audits the official archive, preserves acquisition-file groups, and creates the local dataset used by the experiments.
- Check the UCI page for the dataset's current citation and terms of use.

The experiment scripts record hashes and local metadata when datasets are prepared. Those generated metadata files may contain machine-specific paths and are intentionally excluded from this repository.
