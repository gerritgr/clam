<p align="center">
  <img src="clam_logo.jpg" alt="CLAM logo" width="100">
</p>

# CLAM: Causal Spatial Disaggregation to Infer Local Effects From Coarse Data

<p align="center">
  <img src="motivating_example.jpg" alt="CLAM motivating example" width="600">
</p>

## Overview

**CLAM** estimates fine-grained causal effects from coarse-resolution interventions and outcomes by exploiting high-resolution contextual covariates that modulate these effects.

Many real-world interventions are applied and measured at broad spatial scales, while their effects vary locally. CLAM addresses this mismatch by jointly learning the local causal mechanism and a disaggregation mapping, which enables:

- estimation of local treatment effects from aggregated outcomes,
- counterfactual reasoning under hypothetical interventions,
- disaggregation of outcomes from coarse to fine spatial resolution,
- extensions to settings with unknown aggregation functions, latent intervention locations, or hidden confounders.

This is particularly relevant in domains such as **public health**, **environmental policy**, **education**, and the **social sciences**, where decisions are made at coarse spatial scales despite substantial local heterogeneity.

All experiments are provided as Jupyter notebooks. They run off-the-shelf in Google Colab, or locally with `uv` (see [Running Locally with uv](#running-locally-with-uv)).

## Running the Synthetic Experiments

The synthetic experiments from the paper live in a single notebook:

[**Open the synthetic experiments in Colab**](https://colab.research.google.com/github/gerritgr/clam/blob/main/synthetic_experiments.ipynb)

It covers:

- political campaigning with effects that vary by demographics,
- unknown intervention locations,
- hidden spatial confounders,
- unknown aggregation functions,
- confounded treatment assignment.

Open the link and run the cells top to bottom; everything is self-contained.

## Running the Semi-Synthetic Experiment

The semi-synthetic case study on heat and gun violence is in a separate notebook:

[**Open the semi-synthetic experiment in Colab**](https://colab.research.google.com/github/gerritgr/clam/blob/main/real_world_data_experiment/realworld_experiment.ipynb)

It builds on compiled real-world data (roughly **800 MB** as a CSV). The dataset is split into multi-part zip archives under `real_world_data_experiment/`; the first cell of the notebook locates and unpacks them automatically, or fetches them if they are not present locally.

If you use this data, please cite the original sources alongside our paper:

- [Gun Violence Archive](https://www.gunviolencearchive.org/)
- [AlphaEarth Foundations](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/)

## Additional Experiments

The `backup_experiments/` folder contains supplementary notebooks:

- `exp1_baseline_bicubic.ipynb` — bicubic interpolation baseline for Experiment 1,
- `exp1_baseline_goodman.ipynb` — Goodman regression baseline for Experiment 1,
- `exp1_ablation_hidden_confounder.ipynb` — ablation study with a hidden confounder.

## Running Locally with uv

Dependencies are managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml` declares them and `uv.lock` pins exact versions for a fully reproducible environment.

```bash
# 1. Install uv (fast Python package manager) - skip if already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install dependencies
git clone https://github.com/gerritgr/clam.git
cd clam
uv sync

# 3. Launch JupyterLab
uv run jupyter lab
```

The semi-synthetic notebook unpacks the split zip archives in `real_world_data_experiment/` into the ~800 MB CSV on its first cell, so no manual download is needed.

### One-Click Reproduction

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync && uv run jupyter nbconvert --to python synthetic_experiments.ipynb backup_experiments/*.ipynb real_world_data_experiment/*.ipynb && for f in synthetic_experiments.py backup_experiments/*.py real_world_data_experiment/*.py; do (cd "$(dirname "$f")" && uv run python "$(basename "$f")") || break; done
```

This installs the environment, converts every notebook to a script, and runs them in order, each from its own folder so relative paths resolve. The `|| break` stops the chain on the first failure. Figures are written to disk; `plt.show()` is only called when the notebooks run in Colab, so nothing blocks on a window. The generated `.py` files are gitignored.

Expect the semi-synthetic experiment to take a while; a GPU is used automatically if `torch` finds one.

## Citation

If you use CLAM, the notebooks, or the compiled real-world data, please cite:

```bibtex
@article{grossmann2026clam,
  title={CLAM: Causal Spatial Disaggregation to Infer Local Effects From Coarse Data},
  author={Gro{\ss}mann, Gerrit and Mukherjee, Sumantrak and Vollmer, Sebastian J},
  journal={arXiv preprint arXiv:2608.08064},
  year={2026}
}
```

Please also cite the original data sources listed above when using the real-world data.
