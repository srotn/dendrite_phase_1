# Figure Style

The seven statistical figures use a local, paper-width adaptation of the `science` Matplotlib style from [SciencePlots](https://github.com/garrettj403/SciencePlots), released under the MIT license. The upstream style provides a scientific-paper baseline; this adaptation sets figure dimensions for the manuscript text width, increases typography, and loads Latin Modern through the local TeX Live installation to match the paper.

The source style was checked against SciencePlots 2.1.1 (repository revision `b9b16959570bd2fbc9ff5118bacc423c3bddd592`). Its license is included in `SCIENCEPLOTS_LICENSE.txt`. The plotting script reads the local `.mplstyle` file directly, so importing the upstream package is not required.

Run from the repository root with:

```powershell
python experiments/dendrite/phase_6_publication_strengthening/paper_latex_nn/figures/regenerate_figures.py
```

The script reads only the audited CSV/JSON summaries under `phase_6_publication_strengthening` and writes vector PDF figures to this directory. It does not retrain models or modify experiment records.
