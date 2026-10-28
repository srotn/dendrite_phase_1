# Phase 6: Publication-Strengthening Studies

The four sequential studies add independent IID/OOD evaluation and grouped real-data folds, parameter-matched MLP and classical baselines, continuous training-budget comparisons, and constant-slot/moment-matching/field-subset controls. Their frozen settings are encoded in `common.py` and `protocol.json` generated at run time.

After preparing the Phase 4 datasets, run:

```bash
python experiments/dendrite/phase_6_publication_strengthening/prepare.py
python experiments/dendrite/phase_6_publication_strengthening/verify_engine.py
python experiments/dendrite/phase_6_publication_strengthening/pipeline.py
```

`pipeline.py` executes stages 1–4 in order, verifies their completion markers, runs the locked synthetic confirmation, and writes the final report. The first run writes local protocol locks and source hashes; these runtime files are intentionally not tracked. Training requires a CUDA-capable GPU. Real-data folds are grouped by concrete formulation and CO acquisition file.
