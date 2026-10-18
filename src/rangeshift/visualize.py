from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

CMAP = "viridis"
FONT = 9
DPI = 300

METHOD_COLORS = {
    "learned_cost_ot": "#0072B2",
    "persistence": "#E69F00",
    "sdm_only": "#009E73",
    "uniform_cost_ot": "#CC79A7",
    "climate_velocity": "#D55E00",
}
METHOD_LABELS = {
    "learned_cost_ot": "learned cost",
    "persistence": "persistence",
    "sdm_only": "SDM only",
    "uniform_cost_ot": "uniform-cost OT",
    "climate_velocity": "climate velocity",
}


def _style_map(ax) -> None:
    ax.set_xlabel("x (km)", fontsize=FONT)
    ax.set_ylabel("y (km)", fontsize=FONT)
    ax.set_aspect("equal")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors="0.4", labelsize=FONT)


def plot_cost_surface(mean_cost: np.ndarray, coords_m: np.ndarray) -> Figure:
    coords = np.asarray(coords_m, dtype=np.float64) / 1000.0
    fig = Figure(figsize=(6.5, 5.5), dpi=DPI)
    ax = fig.add_subplot(111)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mean_cost, cmap=CMAP, s=22, marker="s", linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("mean cost to all other cells", fontsize=FONT)
    cbar.ax.tick_params(labelsize=FONT)
    cbar.outline.set_visible(False)
    _style_map(ax)
    fig.tight_layout()
    return fig


def plot_species_panel(
    mu_t0: np.ndarray, mu_t1_sdm: np.ndarray, mu_t1_pred: np.ndarray, mu_t1_obs: np.ndarray,
    coords_m: np.ndarray,
) -> Figure:
    coords = np.asarray(coords_m, dtype=np.float64) / 1000.0
    panels = [
        (r"$\mu_{t_0}$" + "\nobserved, 1997-2002", mu_t0),
        (r"$\mu_{t_1}^{sdm}$" + "\nclimate-feasible", mu_t1_sdm),
        (r"$\mu_{t_1}^{pred}$" + "\nforecast", mu_t1_pred),
        (r"$\mu_{t_1}^{obs}$" + "\nobserved, 2017-2022", mu_t1_obs),
    ]
    pooled = np.concatenate([np.asarray(m)[np.asarray(m) > 0] for _, m in panels])
    vmax = float(np.percentile(pooled, 98))

    fig = Figure(figsize=(15, 4), dpi=DPI)
    axes = fig.subplots(1, 4)
    for ax, (title, mu) in zip(axes, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.asarray(mu), cmap=CMAP,
                        vmin=0, vmax=vmax, s=14, marker="s", linewidths=0)
        ax.set_title(title, fontsize=FONT + 1, color="0.15")
        _style_map(ax)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.015, aspect=25)
    cbar.set_label("probability mass per cell", fontsize=FONT)
    cbar.ax.tick_params(labelsize=FONT)
    cbar.outline.set_visible(False)
    return fig


def plot_loss_curve(losses: list[float]) -> Figure:
    fig = Figure(figsize=(6.5, 4.2), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.plot(range(len(losses)), losses, color="#1f6f5c", linewidth=1.8)
    ax.set_xlabel("epoch", fontsize=FONT)
    ax.set_ylabel("training loss", fontsize=FONT)
    ax.tick_params(labelsize=FONT)
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_baseline_comparison(rows: list[dict], aous: list[str], names: dict[str, str]) -> Figure:
    metrics = [
        ("wasserstein_m", "Wasserstein distance (m), lower is better"),
        ("centroid_shift_m", "Centroid-shift error (m), lower is better"),
        ("northern_half_mass_error", "Northern-half-mass error, lower is better"),
        ("schoeners_d", "Schoener's D, higher is better"),
    ]
    by = {(r["aou"], r["method"]): r for r in rows}
    methods = list(METHOD_COLORS)
    x = np.arange(len(aous))
    width = 0.8 / len(methods)

    fig = Figure(figsize=(13, 9), dpi=DPI)
    axes = fig.subplots(2, 2).ravel()
    for ax, (key, label) in zip(axes, metrics):
        for i, method in enumerate(methods):
            vals = [by[(a, method)][key] for a in aous]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width * 0.92,
                   color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels([names[a] for a in aous], rotation=20, ha="right", fontsize=FONT)
        ax.set_ylabel(label, fontsize=FONT)
        ax.tick_params(labelsize=FONT)
        ax.grid(True, axis="y", color="0.9", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=FONT)
    fig.tight_layout()
    return fig


def plot_signed_bias(bias: dict[str, dict[str, float]]) -> Figure:
    """bias maps species name to {persistence, sdm_only, learned_cost_ot} signed northern-half-mass bias."""
    names = list(bias)
    fig = Figure(figsize=(7.5, 5), dpi=DPI)
    ax = fig.add_subplot(111)
    for y, name in enumerate(names):
        b = bias[name]
        ax.plot([b["persistence"], b["sdm_only"]], [y, y], color="0.75", linewidth=1.5, zorder=1)
        for method, size, marker, z in (("persistence", 70, "o", 3), ("sdm_only", 70, "o", 3), ("learned_cost_ot", 90, "D", 4)):
            ax.scatter([b[method]], [y], color=METHOD_COLORS[method], s=size, marker=marker, zorder=z,
                       label=METHOD_LABELS[method] if y == 0 else None)
    ax.axvline(0, color="0.2", linewidth=1.2, linestyle="--", zorder=0)
    ax.set_ylim(-1.6, len(names) - 0.5)
    ax.text(0, -1.05, "truth (zero bias)", fontsize=FONT, color="0.25", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", linewidth=0.7))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=FONT)
    ax.tick_params(axis="x", labelsize=FONT)
    ax.set_xlabel(r"signed northern-half-mass bias (prediction $-$ observed; positive = too far north)", fontsize=FONT)
    ax.legend(loc="lower right", frameon=False, fontsize=FONT)
    ax.grid(True, axis="x", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_ratio_profile(r: list[float], loss: list[float]) -> Figure:
    fig = Figure(figsize=(7, 5), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.plot(r, loss, "o-", color="#0072B2", linewidth=1.8, markersize=6)
    best = int(np.argmin(loss))
    ax.axvline(r[best], color="#D55E00", linestyle="--", linewidth=1.2, label=f"minimum at r = {r[best]}")
    ax.set_xlabel(r"fixed ratio $\beta_1 / \beta_2$", fontsize=FONT)
    ax.set_ylabel("inner-validation Wasserstein (m)", fontsize=FONT)
    ax.tick_params(labelsize=FONT)
    ax.legend(loc="best", frameon=False, fontsize=FONT)
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
