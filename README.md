# A calibration test for evaluating set-based epistemic uncertainty representations
This repository contains the code to reproduce experiments from the paper 
[A calibration test for evaluating set-based epistemic uncertainty representations](https://arxiv.org/abs/2502.16299).

Authors: Mira Juergens, Thomas Mortier, Viktor Bengs, Eyke Hüllermeier, Willem Waegeman.

## Abstract:

The accurate representation of epistemic uncertainty is a challenging yet essential task in machine learning. A widely used representation corresponds to convex sets of probabilistic predictors, also known as credal sets. One popular way of constructing these credal sets is via ensembling or specialized supervised learning methods, where the epistemic uncertainty can be quantified through measures such as the set size or the disagreement among members. In principle, these sets should contain the true data-generating distribution. As a necessary condition for this validity, we adopt the strongest notion of calibration as a proxy. Concretely, we propose a novel statistical test to determine whether there is a convex combination of the set's predictions that is calibrated in distribution. In contrast to previous methods, our framework allows the convex combination to be instance dependent, recognizing that different ensemble members may be better calibrated in different regions of the input space. Moreover, we learn this combination via proper scoring rules, which inherently optimize for calibration. Building on differentiable, kernel-based estimators of calibration errors, we introduce a nonparametric testing procedure and demonstrate the benefits of capturing instance-level variability on of synthetic and real-world experiments.

## 🛠️ Installation

Everything is pure-Python and published on PyPI, so a single
`pip install -e .` is enough.  
We recommend a fresh virtual environment (conda, venv, mamba, …).

```bash
git clone https://github.com/mkjuergens/EnsembleCalibration.git
cd EnsembleCalibration

# optional – create / activate a clean env first
# conda create -n enscal python=3.11
# conda activate enscal

pip install -e .

```

## Run synthetic experiment
All experiments based on synthetic data can be run via 
```bash
python src/main_t1t2_synthetic.py          
```
where a timestamped folder with the results will be created automatically:
```bash
results/
└── <experiment>/          # gp or dirichlet
    └── <bootstrap_test>/  # v0, v2, v_base, …
        └── 2025-05-31-14-30-25/
            ├── h0_const.csv     # Type-I: λ=const
            ├── h0_xdep.csv      # Type-I: λ=f(x)
            ├── h1_<δ>.csv       # one per alternative
            ├── full_results.csv # concatenated view
            ├── error_analysis_t1.png
            ├── error_analysis_t2.png
            └── config.json      # full CLI args
```