# -*- coding: utf-8 -*-
"""
Single-cell Colab experiment:

Original method:
    Learn subregional response parameters from only regional outcome loss.

Baseline:
    Naively disaggregate each regional outcome uniformly to all subregions,
    then train with one loss signal per subregion.

Outputs:
    data_exp1/
        estimation_loss.csv
        estimation_loss_baseline.csv
        estimation_loss_both.csv
        estimation_loss.jpg
        estimation_loss.pdf
        estimation_loss_baseline.jpg
        estimation_loss_baseline.pdf
        estimation_loss_both.jpg
        estimation_loss_both.pdf
        parameter_estimate_method.txt
        parameter_estimate_baseline.txt
        causal_effect_subregion_estimated_method.csv
        causal_effect_subregion_estimated_baseline.csv
        outcome_subregion_estimated_method.csv
        outcome_subregion_estimated_baseline.csv
        all_outputs_exp1.zip
"""

# ============================================================
# Imports
# ============================================================

import os
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn


# ============================================================
# Parameters and hyperparameters
# ============================================================

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42

# ----------------------------
# Data parameters
# ----------------------------
data_dir = "data_exp1"

num_regions = 100
subregions_per_region = 100

region_grid_size = 10
sub_grid_size = 100
sub_per_region_side = 10

# Controls the random imbalance between poor and rich subregions.
SPATIAL_VARIANCE = 5

# Noise variance for subregional observed outcomes.
noise_variance = 0.01

# ----------------------------
# True causal effects
# ----------------------------
# The true treatment effect depends on wealth/context.
TRUE_EFFECT_POOR = -0.1
TRUE_EFFECT_MIDDLE = 0.0
TRUE_EFFECT_RICH = 0.3

# ----------------------------
# Training parameters
# ----------------------------
n_epochs = 1000
lr = 0.1

# If True, use log-log axes for loss plots.
loglog_flag = True

# ----------------------------
# Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Plot parameters
# ----------------------------
label_fontsize = 20
tick_fontsize = 18
title_fontsize = 22
legend_fontsize = 14
line_width = 3

os.makedirs(data_dir, exist_ok=True)


# ============================================================
# Proper seeding
# ============================================================

def set_all_seeds(seed):
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


set_all_seeds(SEED)

print("Using device:", device)
print("Seed:", SEED)


# ============================================================
# Visual style
# ============================================================

colors = sns.color_palette("deep")

blue_color = colors[0]
red_color = colors[3]

red_cmap = sns.light_palette(colors[3], as_cmap=True)
blue_cmap = sns.light_palette(colors[0], as_cmap=True)
green_cmap = sns.light_palette(colors[2], as_cmap=True)
purple_cmap = sns.light_palette(colors[4], as_cmap=True)
orange_cmap = sns.light_palette(colors[1], as_cmap=True)


# ============================================================
# Reshaping utilities
# ============================================================

def reshape_to_reggrid(data):
    """
    Convert a vector of length 100 into a 10x10 regional grid.
    """
    return np.asarray(data).reshape(region_grid_size, region_grid_size)


def reshape_to_subgrid(data):
    """
    Convert a 100x100 matrix into a 100x100 spatial grid.

    Each row is one region.
    Each region row contains 100 subregions, interpreted as a 10x10 block.
    """
    data = np.asarray(data)
    grid = np.zeros((sub_grid_size, sub_grid_size))

    for ri in range(region_grid_size):
        for rj in range(region_grid_size):
            reg_idx = ri * region_grid_size + rj
            sub_data = data[reg_idx].reshape(sub_per_region_side, sub_per_region_side)

            row_start = ri * sub_per_region_side
            row_end = (ri + 1) * sub_per_region_side

            col_start = rj * sub_per_region_side
            col_end = (rj + 1) * sub_per_region_side

            grid[row_start:row_end, col_start:col_end] = sub_data

    return grid


# ============================================================
# Data generation
# ============================================================

# 1) Intervention matrix: 100x100.
# Each row is either all 0 or all 1.
interventions_region_flags = np.random.choice(
    [0, 1],
    size=num_regions,
    p=[0.5, 0.5],
)

interventions_sub = np.repeat(
    interventions_region_flags[:, np.newaxis],
    subregions_per_region,
    axis=1,
)

interventions_reg = interventions_sub.mean(axis=1)

pd.DataFrame(interventions_sub).to_csv(
    os.path.join(data_dir, "interventions_subregion.csv"),
    index=False,
    header=False,
)

pd.DataFrame(interventions_reg).to_csv(
    os.path.join(data_dir, "interventions_region.csv"),
    index=False,
    header=False,
)


# 2) Context matrix: 100x100, wealth levels 1, 2, 3.
context_sub = np.zeros((num_regions, subregions_per_region), dtype=int)

for i in range(num_regions):
    if i == 0:
        # First row: all middle class.
        context_sub[i, :] = 2

    elif i == 1:
        # Second row: exactly half poor and half rich.
        half = subregions_per_region // 2
        arr = np.concatenate([
            np.ones(half, dtype=int),
            np.full(subregions_per_region - half, 3, dtype=int),
        ])
        np.random.shuffle(arr)
        context_sub[i, :] = arr

    else:
        # Other rows:
        # randomly choose number of middle-class subregions.
        num_middle = np.random.randint(20, 101)
        remaining = subregions_per_region - num_middle

        deviation = np.random.randint(-SPATIAL_VARIANCE, SPATIAL_VARIANCE + 1)

        num_poor = max(
            0,
            min(
                remaining,
                remaining // 2 + deviation,
            ),
        )
        num_rich = remaining - num_poor

        arr = np.concatenate([
            np.ones(num_poor, dtype=int),
            np.full(num_middle, 2, dtype=int),
            np.full(num_rich, 3, dtype=int),
        ])

        np.random.shuffle(arr)
        context_sub[i, :] = arr

context_reg = context_sub.mean(axis=1)

pd.DataFrame(context_sub).to_csv(
    os.path.join(data_dir, "context_subregion.csv"),
    index=False,
    header=False,
)

pd.DataFrame(context_reg).to_csv(
    os.path.join(data_dir, "context_region.csv"),
    index=False,
    header=False,
)


# 3) Noise matrix.
noise_sub = np.random.normal(
    loc=0.0,
    scale=np.sqrt(noise_variance),
    size=(num_regions, subregions_per_region),
)

pd.DataFrame(noise_sub).to_csv(
    os.path.join(data_dir, "noise_subregion.csv"),
    index=False,
    header=False,
)


# 4) True causal effect matrix.
causal_effect_sub = np.zeros((num_regions, subregions_per_region), dtype=float)
causal_effect_sub[context_sub == 1] = TRUE_EFFECT_POOR
causal_effect_sub[context_sub == 2] = TRUE_EFFECT_MIDDLE
causal_effect_sub[context_sub == 3] = TRUE_EFFECT_RICH

pd.DataFrame(causal_effect_sub).to_csv(
    os.path.join(data_dir, "causal_effect_subregion.csv"),
    index=False,
    header=False,
)


# 5) Outcome matrix.
# outcome = noise + intervention * causal_effect
outcome_sub = noise_sub + interventions_sub * causal_effect_sub
outcome_reg = outcome_sub.mean(axis=1)

pd.DataFrame(outcome_sub).to_csv(
    os.path.join(data_dir, "outcome_subregion.csv"),
    index=False,
    header=False,
)

pd.DataFrame(outcome_reg).to_csv(
    os.path.join(data_dir, "outcome_region.csv"),
    index=False,
    header=False,
)


# ============================================================
# Naive uniform disaggregation baseline target
# ============================================================

# Uniform assumption:
# Every subregion inside region i receives the same target value:
#     outcome_sub_uniform[i, j] = outcome_reg[i]
#
# This gives a micro-level loss signal for every element, but it is
# based on a deliberately naive disaggregation assumption.
outcome_sub_uniform = np.repeat(
    outcome_reg[:, np.newaxis],
    subregions_per_region,
    axis=1,
)

pd.DataFrame(outcome_sub_uniform).to_csv(
    os.path.join(data_dir, "outcome_subregion_uniform_baseline_target.csv"),
    index=False,
    header=False,
)


# ============================================================
# Optional heatmaps for visual inspection
# ============================================================

def plot_heatmap(grid, cmap, title, filename, figsize, vmin=None, vmax=None, linewidths=0, cbar=True):
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        grid,
        ax=ax,
        cmap=cmap,
        square=True,
        linewidths=linewidths,
        cbar=cbar,
        cbar_kws={"location": "right"} if cbar else None,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    if cbar:
        cbar_obj = ax.collections[0].colorbar
        cbar_obj.ax.tick_params(labelsize=14)

    fig.savefig(os.path.join(data_dir, f"{filename}.jpg"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(data_dir, f"{filename}.pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


plot_heatmap(
    reshape_to_reggrid(interventions_reg),
    red_cmap,
    "Interventions Region",
    "interventions_region",
    figsize=(5, 5),
    vmin=0,
    vmax=1,
    linewidths=1,
)

plot_heatmap(
    reshape_to_reggrid(outcome_reg),
    blue_cmap,
    "Outcome Region",
    "outcome_region",
    figsize=(5, 5),
    vmin=-0.05,
    vmax=0.1,
    linewidths=1,
)

plot_heatmap(
    reshape_to_subgrid(context_sub),
    green_cmap,
    "Context Subregion",
    "context_subregion",
    figsize=(10, 10),
    vmin=1,
    vmax=3,
    linewidths=0,
)

plot_heatmap(
    reshape_to_subgrid(causal_effect_sub),
    purple_cmap,
    "True Causal Effect Subregion",
    "causal_effect_subregion",
    figsize=(10, 10),
    vmin=-0.1,
    vmax=0.3,
    linewidths=0,
)


# ============================================================
# PyTorch tensors
# ============================================================

interventions_reg_t = torch.tensor(
    interventions_reg,
    dtype=torch.long,
    device=device,
)

context_sub_t = torch.tensor(
    context_sub,
    dtype=torch.long,
    device=device,
)

outcome_reg_t = torch.tensor(
    outcome_reg,
    dtype=torch.float32,
    device=device,
)

outcome_sub_uniform_t = torch.tensor(
    outcome_sub_uniform,
    dtype=torch.float32,
    device=device,
)

causal_effect_true_t = torch.tensor(
    causal_effect_sub,
    dtype=torch.float32,
    device=device,
)


# ============================================================
# Model
# ============================================================

class OutcomeModel(nn.Module):
    """
    Six-parameter response model.

    theta[0] = f(t=0, c=1)
    theta[1] = f(t=0, c=2)
    theta[2] = f(t=0, c=3)

    theta[3] = f(t=1, c=1)
    theta[4] = f(t=1, c=2)
    theta[5] = f(t=1, c=3)
    """

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.ones(6))

    def forward(self, interventions_reg_t, context_sub_t):
        pred_sub = torch.zeros_like(context_sub_t, dtype=torch.float32)

        theta = self.theta

        mask_t0 = interventions_reg_t[:, None] == 0
        mask_t1 = interventions_reg_t[:, None] == 1

        pred_sub[mask_t0 & (context_sub_t == 1)] = theta[0]
        pred_sub[mask_t0 & (context_sub_t == 2)] = theta[1]
        pred_sub[mask_t0 & (context_sub_t == 3)] = theta[2]

        pred_sub[mask_t1 & (context_sub_t == 1)] = theta[3]
        pred_sub[mask_t1 & (context_sub_t == 2)] = theta[4]
        pred_sub[mask_t1 & (context_sub_t == 3)] = theta[5]

        pred_reg = pred_sub.mean(dim=1)

        return pred_reg, pred_sub


def estimate_current_causaleffectmatrix(model, context_sub_t):
    """
    Estimate causal effect:
        f(1, c) - f(0, c)
    """
    causal_effect_sub_t = torch.zeros_like(context_sub_t, dtype=torch.float32)

    causal_effect_sub_t[context_sub_t == 1] = model.theta[3] - model.theta[0]
    causal_effect_sub_t[context_sub_t == 2] = model.theta[4] - model.theta[1]
    causal_effect_sub_t[context_sub_t == 3] = model.theta[5] - model.theta[2]

    return causal_effect_sub_t


# ============================================================
# Training: given method
# ============================================================

def train_given_method():
    """
    Given method:
        Predict all subregional outcomes.
        Aggregate predictions to regional means.
        Compute loss only on regional outcomes.
    """
    set_all_seeds(SEED)

    model = OutcomeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    ce_loss_hist = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_reg, pred_sub = model(interventions_reg_t, context_sub_t)

        # Regional-only loss.
        loss = torch.mean((pred_reg - outcome_reg_t) ** 2)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            causal_effect_est_t = estimate_current_causaleffectmatrix(model, context_sub_t)
            ce_mse = torch.mean((causal_effect_est_t - causal_effect_true_t) ** 2)

        loss_hist.append(loss.item())
        ce_loss_hist.append(ce_mse.item())

    return model, np.array(loss_hist), np.array(ce_loss_hist)


# ============================================================
# Training: baseline with naive uniform disaggregation
# ============================================================

def train_uniform_baseline():
    """
    Baseline:
        First disaggregate each regional outcome uniformly:
            outcome_sub_uniform[i, j] = outcome_reg[i]

        Then train using element-wise subregional loss.

    This gives one loss signal per element, but the target is biased
    because the uniform assumption removes within-region heterogeneity.
    """
    set_all_seeds(SEED)

    model = OutcomeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    ce_loss_hist = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_reg, pred_sub = model(interventions_reg_t, context_sub_t)

        # Element-wise loss against uniformly disaggregated target.
        loss = torch.mean((pred_sub - outcome_sub_uniform_t) ** 2)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            causal_effect_est_t = estimate_current_causaleffectmatrix(model, context_sub_t)
            ce_mse = torch.mean((causal_effect_est_t - causal_effect_true_t) ** 2)

        loss_hist.append(loss.item())
        ce_loss_hist.append(ce_mse.item())

    return model, np.array(loss_hist), np.array(ce_loss_hist)


print("\nTraining given method...")
model_method, loss_method, ce_loss_method = train_given_method()

print("Training uniform-disaggregation baseline...")
model_baseline, loss_baseline, ce_loss_baseline = train_uniform_baseline()


# ============================================================
# Save losses
# ============================================================

pd.DataFrame({
    "epoch": np.arange(1, n_epochs + 1),
    "loss": loss_method,
    "ce_loss": ce_loss_method,
}).to_csv(
    os.path.join(data_dir, "estimation_loss.csv"),
    index=False,
)

pd.DataFrame({
    "epoch": np.arange(1, n_epochs + 1),
    "loss": loss_baseline,
    "ce_loss": ce_loss_baseline,
}).to_csv(
    os.path.join(data_dir, "estimation_loss_baseline.csv"),
    index=False,
)

pd.DataFrame({
    "epoch": np.arange(1, n_epochs + 1),
    "method_training_loss": loss_method,
    "method_causal_effect_mse": ce_loss_method,
    "baseline_training_loss": loss_baseline,
    "baseline_causal_effect_mse": ce_loss_baseline,
}).to_csv(
    os.path.join(data_dir, "estimation_loss_both.csv"),
    index=False,
)


# ============================================================
# Plot: estimation_loss.jpg / .pdf for given method
# ============================================================

epochs = np.arange(1, n_epochs + 1)

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("Epoch", fontsize=label_fontsize)
ax1.set_ylabel("Training Loss", fontsize=label_fontsize, color=blue_color)

if loglog_flag:
    ax1.loglog(epochs, loss_method, color=blue_color, alpha=0.9, lw=3)
else:
    ax1.plot(epochs, loss_method, color=blue_color, alpha=0.9, lw=3)

ax1.tick_params(
    axis="both",
    which="both",
    length=0,
    labelsize=tick_fontsize,
    colors=blue_color,
)

ax1.set_title("MSE Curves", fontsize=title_fontsize)
ax1.spines["top"].set_visible(False)

ax2 = ax1.twinx()
ax2.set_ylabel("MSE Causal Effect", fontsize=label_fontsize, color=red_color)

if loglog_flag:
    ax2.loglog(epochs, ce_loss_method, color=red_color, alpha=0.9, ls="--", lw=5)
else:
    ax2.plot(epochs, ce_loss_method, color=red_color, alpha=0.9, ls="--", lw=5)

ax2.tick_params(
    axis="y",
    which="both",
    length=0,
    labelsize=tick_fontsize,
    colors=red_color,
)

ax2.spines["top"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(data_dir, "estimation_loss.jpg"), dpi=300)
fig.savefig(os.path.join(data_dir, "estimation_loss.pdf"))
plt.show()
plt.close(fig)


# ============================================================
# Plot: estimation_loss_baseline.jpg / .pdf
# Same visual structure, but baseline curves are black.
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("Epoch", fontsize=label_fontsize)
ax1.set_ylabel("Training Loss", fontsize=label_fontsize, color="black")

if loglog_flag:
    ax1.loglog(epochs, loss_baseline, color="black", alpha=0.30, lw=3)
else:
    ax1.plot(epochs, loss_baseline, color="black", alpha=0.30, lw=3)

ax1.tick_params(
    axis="both",
    which="both",
    length=0,
    labelsize=tick_fontsize,
    colors="black",
)

ax1.set_title("MSE Curves: Uniform Baseline", fontsize=title_fontsize)
ax1.spines["top"].set_visible(False)

ax2 = ax1.twinx()
ax2.set_ylabel("MSE Causal Effect", fontsize=label_fontsize, color="black")

if loglog_flag:
    ax2.loglog(epochs, ce_loss_baseline, color="black", alpha=0.50, ls="--", lw=5)
else:
    ax2.plot(epochs, ce_loss_baseline, color="black", alpha=0.50, ls="--", lw=5)

ax2.tick_params(
    axis="y",
    which="both",
    length=0,
    labelsize=tick_fontsize,
    colors="black",
)

ax2.spines["top"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(data_dir, "estimation_loss_baseline.jpg"), dpi=300)
fig.savefig(os.path.join(data_dir, "estimation_loss_baseline.pdf"))
plt.show()
plt.close(fig)


# ============================================================
# Plot: estimation_loss_both.jpg / .pdf
# Contains both the given method and the baseline.
# Baseline lines are black with 30% and 50% opacity.
# ============================================================
# ============================================================
# Plot: estimation_loss_both.jpg / .pdf
# Contains:
#   - given method training loss, blue solid
#   - given method causal-effect MSE, red dashed
#   - baseline causal-effect MSE only, black dashed with 50% opacity
# No legend.
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("Epoch", fontsize=label_fontsize)
ax1.set_ylabel("Training Loss", fontsize=label_fontsize, color=blue_color)

if loglog_flag:
    ax1.loglog(
        epochs,
        loss_method,
        color=blue_color,
        alpha=0.9,
        lw=3,
    )
else:
    ax1.plot(
        epochs,
        loss_method,
        color=blue_color,
        alpha=0.9,
        lw=3,
    )

ax1.tick_params(
    axis="both",
    which="both",
    length=0,
    labelsize=tick_fontsize,
    colors=blue_color,
)

ax1.set_title("MSE Curves", fontsize=title_fontsize)
ax1.spines["top"].set_visible(False)

ax2 = ax1.twinx()
ax2.set_ylabel("MSE Causal Effect", fontsize=label_fontsize, color=red_color)

if loglog_flag:
    ax2.loglog(
        epochs,
        ce_loss_method,
        color=red_color,
        alpha=0.9,
        ls="--",
        lw=5,
    )

    ax2.loglog(
        epochs,
        ce_loss_baseline,
        color="black",
        alpha=0.50,
        ls="--",
        lw=5,
    )
else:
    ax2.plot(
        epochs,
        ce_loss_method,
        color=red_color,
        alpha=0.9,
        ls="--",
        lw=5,
    )

    ax2.plot(
        epochs,
        ce_loss_baseline,
        color="black",
        alpha=0.50,
        ls="--",
        lw=5,
    )

ax2.tick_params(
    axis="y",
    which="both",
    length=0,
    labelsize=tick_fontsize,
    colors=red_color,
)

ax2.spines["top"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(data_dir, "estimation_loss_both.jpg"), dpi=300)
fig.savefig(os.path.join(data_dir, "estimation_loss_both.pdf"))
plt.show()
plt.close(fig)


# ============================================================
# Save parameter estimates
# ============================================================

theta_method = model_method.theta.detach().cpu().numpy()
theta_baseline = model_baseline.theta.detach().cpu().numpy()

with open(os.path.join(data_dir, "parameter_estimate_method.txt"), "w") as f:
    f.write("theta1,theta2,theta3,theta4,theta5,theta6\n")
    f.write(",".join(map(str, theta_method.flatten())) + "\n")

with open(os.path.join(data_dir, "parameter_estimate_baseline.txt"), "w") as f:
    f.write("theta1,theta2,theta3,theta4,theta5,theta6\n")
    f.write(",".join(map(str, theta_baseline.flatten())) + "\n")


# ============================================================
# Save estimated subregional outcomes and causal effects
# ============================================================

with torch.no_grad():
    _, outcome_sub_est_method_t = model_method(interventions_reg_t, context_sub_t)
    causal_effect_est_method_t = estimate_current_causaleffectmatrix(model_method, context_sub_t)

    _, outcome_sub_est_baseline_t = model_baseline(interventions_reg_t, context_sub_t)
    causal_effect_est_baseline_t = estimate_current_causaleffectmatrix(model_baseline, context_sub_t)

outcome_sub_est_method = outcome_sub_est_method_t.detach().cpu().numpy()
causal_effect_est_method = causal_effect_est_method_t.detach().cpu().numpy()

outcome_sub_est_baseline = outcome_sub_est_baseline_t.detach().cpu().numpy()
causal_effect_est_baseline = causal_effect_est_baseline_t.detach().cpu().numpy()

pd.DataFrame(outcome_sub_est_method).to_csv(
    os.path.join(data_dir, "outcome_subregion_estimated_method.csv"),
    index=False,
    header=False,
)

pd.DataFrame(causal_effect_est_method).to_csv(
    os.path.join(data_dir, "causal_effect_subregion_estimated_method.csv"),
    index=False,
    header=False,
)

pd.DataFrame(outcome_sub_est_baseline).to_csv(
    os.path.join(data_dir, "outcome_subregion_estimated_baseline.csv"),
    index=False,
    header=False,
)

pd.DataFrame(causal_effect_est_baseline).to_csv(
    os.path.join(data_dir, "causal_effect_subregion_estimated_baseline.csv"),
    index=False,
    header=False,
)


# ============================================================
# Visualize estimated causal effect matrices
# ============================================================

plot_heatmap(
    reshape_to_subgrid(causal_effect_est_method),
    purple_cmap,
    "Estimated Causal Effect: Method",
    "causal_effect_subregion_estimated_method",
    figsize=(10, 10),
    vmin=-0.1,
    vmax=0.3,
    linewidths=0,
)

plot_heatmap(
    reshape_to_subgrid(causal_effect_est_baseline),
    purple_cmap,
    "Estimated Causal Effect: Uniform Baseline",
    "causal_effect_subregion_estimated_baseline",
    figsize=(10, 10),
    vmin=-0.1,
    vmax=0.3,
    linewidths=0,
)


# ============================================================
# Print final comparison
# ============================================================

print("\nFinal theta estimates: method")
print("theta1 (t=0, poor):        ", theta_method[0])
print("theta2 (t=0, middle):      ", theta_method[1])
print("theta3 (t=0, rich):        ", theta_method[2])
print("theta4 (t=1, poor):        ", theta_method[3])
print("theta5 (t=1, middle):      ", theta_method[4])
print("theta6 (t=1, rich):        ", theta_method[5])

print("\nFinal theta estimates: uniform baseline")
print("theta1 (t=0, poor):        ", theta_baseline[0])
print("theta2 (t=0, middle):      ", theta_baseline[1])
print("theta3 (t=0, rich):        ", theta_baseline[2])
print("theta4 (t=1, poor):        ", theta_baseline[3])
print("theta5 (t=1, middle):      ", theta_baseline[4])
print("theta6 (t=1, rich):        ", theta_baseline[5])

print("\nFinal losses")
print("Method training loss:       ", loss_method[-1])
print("Method causal effect MSE:   ", ce_loss_method[-1])
print("Baseline training loss:     ", loss_baseline[-1])
print("Baseline causal effect MSE: ", ce_loss_baseline[-1])

