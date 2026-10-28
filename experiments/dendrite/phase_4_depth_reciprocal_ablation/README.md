# Phase 4: Primary Depth Ablation

The primary study compares original, duplicated, and reciprocal input representations across six tasks, six multiplicative depths, and five optimization seeds. Each fit uses 2,000 epochs, Adam with learning rate `2e-4`, batch size 4,096, and gradient-norm clipping at 5.

Prepare source datasets using the scripts and instructions in the repository root, then run:

```bash
python experiments/dendrite/phase_4_depth_reciprocal_ablation/study.py --prepare
python experiments/dendrite/phase_4_depth_reciprocal_ablation/study.py --run-all
python experiments/dendrite/phase_4_depth_reciprocal_ablation/analyze.py
```

Training uses CUDA graphs and requires a CUDA-capable GPU. Generated tensors, checkpoints, metrics, and per-run curves remain local and are ignored by Git. The frozen numerical protocol is in `protocol.json`.
