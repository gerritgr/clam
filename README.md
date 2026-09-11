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

All experiments are provided as Jupyter notebooks that run off-the-shelf in Google Colab — no local setup required.

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
