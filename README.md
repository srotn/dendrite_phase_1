# Dendrite Reciprocal-Input Depth Study

This repository contains the experiment and analysis code associated with the manuscript *Task-Dependent Effects of Reciprocal Inputs Across Dendrite Net Depths*.

## Contents

- The six-task, six-depth primary ablation code from Phase 4.
- The four sequential publication-strengthening studies from Phase 6.
- The synthetic task generators and public-dataset preparation code required by those studies.
- Frozen result summaries, manuscript tables, and the seven vector figures used by the paper.
- A script that regenerates all seven figures from the included aggregate summaries.

The repository excludes raw datasets, prepared tensors, model checkpoints, per-epoch logs, and prediction arrays. The two real-world datasets are available from their cited public sources; their acquisition and preparation steps are documented in [DATA_SOURCES.md](DATA_SOURCES.md). The included aggregate summaries are sufficient to regenerate the paper figures without downloading the datasets or running model training.

## Regenerate the Figures

Use Python 3.13 and install the figure dependencies:

```bash
python -m pip install -r requirements-figures.txt
```

The plotting script uses Matplotlib's LaTeX text rendering to match the manuscript font. Install TeX Live with Latin Modern and make `latex` available on `PATH`, then run:

```bash
python experiments/dendrite/phase_6_publication_strengthening/paper_latex_nn/figures/regenerate_figures.py
```

The generated PDFs are written beside the script.

## Reproduce the Training Experiments

Full training reproduction requires a CUDA-enabled PyTorch installation, substantial GPU time and disk space, and the public datasets described in [DATA_SOURCES.md](DATA_SOURCES.md). Install the analysis dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install a compatible CUDA build of PyTorch separately using the official PyTorch installation selector. The recorded experiment environment used Python 3.13.7, PyTorch 2.12.1+cu132, NumPy 2.4.2, SciPy 1.18.0, pandas 3.0.0, scikit-learn 1.9.0, and CUDA 13.2.

Prepare the inputs and run the primary study in order:

```bash
python scripts/prepare_synthetic_inputs.py
python scripts/prepare_concrete_inputs.py
python experiments/dendrite/phase_3_real_world_regression/01_gas_sensor_temperature_modulation/prepare_data.py
python experiments/dendrite/phase_4_depth_reciprocal_ablation/study.py --prepare
python experiments/dendrite/phase_4_depth_reciprocal_ablation/study.py --run-all
python experiments/dendrite/phase_4_depth_reciprocal_ablation/analyze.py
```

Then prepare the follow-up datasets, verify the training engine, and run the four follow-up stages sequentially:

```bash
python experiments/dendrite/phase_6_publication_strengthening/prepare.py
python experiments/dendrite/phase_6_publication_strengthening/verify_engine.py
python experiments/dendrite/phase_6_publication_strengthening/pipeline.py
```

Run the commands from the repository root. Generated data, checkpoints, logs, and verification locks are excluded by `.gitignore` and remain local. Runtime and small numerical differences can depend on the PyTorch build, GPU, and CUDA libraries.

## License

The experiment code and documentation are released under the MIT License. The SciencePlots style adaptation retains its upstream license notice in the figure directory. Third-party datasets retain the terms specified by their source repositories.
