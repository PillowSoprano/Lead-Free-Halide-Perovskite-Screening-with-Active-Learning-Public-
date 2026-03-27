"""
GP Active Learning Baseline
============================
Implements a Gaussian Process regression + UCB acquisition baseline
for comparison with the ensemble-based active learning in the paper.

Usage:
    python gp_al_baseline.py

This script runs on the same data and uses the same experimental setup
(n0=200, K=50, 1 acquisition round, 10 seeds, group-based test split)
as the existing active_learning_simulation.py, so results are directly
comparable.

Outputs:
    outputs/gp_al_comparison.csv   — per-seed MAE for GP-UCB vs ensemble vs random
    outputs/gp_al_comparison.png   — bar chart comparison
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
from scipy.stats import norm as scipy_norm, ttest_rel

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# GP model wrapper
# ─────────────────────────────────────────────

class GPModel:
    """Gaussian Process regressor with Matern-5/2 kernel."""

    def __init__(self, random_state: int = 42):
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=random_state,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)
        return self

    def predict_with_uncertainty(self, X):
        X_scaled = self.scaler.transform(X)
        mu, sigma = self.gp.predict(X_scaled, return_std=True)
        return mu, sigma

    def predict(self, X):
        mu, _ = self.predict_with_uncertainty(X)
        return mu


# ─────────────────────────────────────────────
# Ensemble model (copied from simulation for fairness)
# ─────────────────────────────────────────────

class EnsembleModel:
    """15-member heterogeneous ensemble (same as paper)."""

    def __init__(self, random_state: int = 42):
        from sklearn.ensemble import GradientBoostingRegressor
        self.members = []
        for i in range(15):
            if i % 3 == 0:
                m = GradientBoostingRegressor(
                    n_estimators=200, max_depth=5,
                    learning_rate=0.05, random_state=random_state + i
                )
            else:
                m = RandomForestRegressor(
                    n_estimators=200, max_depth=20,
                    min_samples_leaf=2, random_state=random_state + i,
                    n_jobs=-1
                )
            self.members.append(m)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        rng = np.random.default_rng(42)
        for m in self.members:
            idx = rng.choice(len(X_scaled), size=len(X_scaled), replace=True)
            m.fit(X_scaled[idx], y[idx])
        return self

    def predict_with_uncertainty(self, X):
        X_scaled = self.scaler.transform(X)
        preds = np.stack([m.predict(X_scaled) for m in self.members])
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)

    def predict(self, X):
        mu, _ = self.predict_with_uncertainty(X)
        return mu


# ─────────────────────────────────────────────
# Acquisition functions
# ─────────────────────────────────────────────

def ucb_acquisition(mu, sigma, k, kappa=2.0, **kwargs):
    """Upper Confidence Bound: score = mu + kappa * sigma."""
    scores = mu + kappa * sigma
    return np.argsort(scores)[-k:]


def ei_acquisition(mu, sigma, k, current_best, xi=0.01, **kwargs):
    """Expected Improvement over current_best."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (mu - current_best - xi) / sigma_safe
    ei = (mu - current_best - xi) * scipy_norm.cdf(z) + sigma_safe * scipy_norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return np.argsort(ei)[-k:]


def uncertainty_acquisition(mu, sigma, k, **kwargs):
    """Pure uncertainty sampling."""
    return np.argsort(sigma)[-k:]


def random_acquisition(mu, sigma, k, rng, **kwargs):
    """Random baseline."""
    return rng.choice(len(mu), size=k, replace=False)


# ─────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────

def run_trial(X, y, groups, model_cls, acq_fn, acq_name,
              n0=200, K=50, seed=0):
    """
    One trial of pool-based active learning.

    Returns dict with:
        mae_before  — MAE on test set after training on n0 points
        mae_after   — MAE on test set after querying K more points
        delta_mae   — relative MAE reduction (%)
    """
    rng = np.random.default_rng(seed)

    # Group-based test split (same as paper)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_pool_idx, test_idx = next(gss.split(X, groups=groups))

    X_pool, y_pool = X[train_pool_idx], y[train_pool_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Initial labeled set
    L_idx = rng.choice(len(X_pool), size=min(n0, len(X_pool)), replace=False)
    U_idx = np.setdiff1d(np.arange(len(X_pool)), L_idx)

    # Train on L
    model = model_cls(random_state=seed)
    model.fit(X_pool[L_idx], y_pool[L_idx])
    mae_before = mean_absolute_error(y_test, model.predict(X_test))

    # Acquire K points from U
    mu_U, sigma_U = model.predict_with_uncertainty(X_pool[U_idx])
    k_actual = min(K, len(U_idx))

    acq_kwargs = dict(
        current_best=float(y_pool[L_idx].max()),
        rng=rng,
    )
    selected_in_U = acq_fn(mu_U, sigma_U, k_actual, **acq_kwargs)
    queried = U_idx[selected_in_U]

    # Retrain on L + queried
    new_L = np.concatenate([L_idx, queried])
    model2 = model_cls(random_state=seed + 1)
    model2.fit(X_pool[new_L], y_pool[new_L])
    mae_after = mean_absolute_error(y_test, model2.predict(X_test))

    delta_mae = (mae_before - mae_after) / mae_before * 100  # % reduction

    return {
        "acquisition": acq_name,
        "seed": seed,
        "mae_before": mae_before,
        "mae_after": mae_after,
        "delta_mae_pct": delta_mae,
    }


# ─────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────

def load_data():
    """
    Load features and band gaps using the same pipeline as the main script.
    Tries cached CSV first; if not found, fetches from Materials Project API
    and extracts compositional features using EnhancedFeatureExtractor.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dotenv import load_dotenv
    load_dotenv()

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    cache_path = os.path.join(outputs_dir, "gp_feature_dataset.csv")

    # ── Try loading from cache ─────────────────────────────────────────────
    if os.path.exists(cache_path):
        print(f"Loading cached feature dataset from {cache_path}")
        df_cache = pd.read_csv(cache_path)
        non_feat = {"formula", "band_gap", "e_above_hull", "Unnamed: 0"}
        feat_cols = [c for c in df_cache.columns if c not in non_feat]
        X = df_cache[feat_cols].values.astype(float)
        y = df_cache["band_gap"].values.astype(float)
        formulas = df_cache["formula"].tolist()
        groups = _get_groups(formulas)
        print(f"Loaded {len(y)} compounds, {X.shape[1]} features.")
        return X, y, groups

    # ── Fetch from MP + featurize ──────────────────────────────────────────
    print("Building feature dataset from Materials Project + local featurizer...")
    from mp_api.client import MPRester
    from improved_perovskite_screening import (
        Config, EnhancedFeatureExtractor, PerovskiteValidator
    )

    api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    if not api_key:
        raise RuntimeError("Set MATERIALS_PROJECT_API_KEY in .env")

    config = Config()
    extractor = EnhancedFeatureExtractor()

    print("  Querying Materials Project...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            exclude_elements=["Pb"],
            num_elements=(config.min_elements, config.max_elements),
            band_gap=(config.bandgap_min, config.bandgap_max),
            fields=["material_id", "formula_pretty", "band_gap",
                    "energy_above_hull", "composition", "elements", "symmetry"]
        )
    print(f"  API returned {len(docs)} materials, applying filters...")

    data, feature_list = [], []
    for doc in docs:
        comp_dict = doc.composition.as_dict()
        sg = doc.symmetry.number if hasattr(doc, "symmetry") and doc.symmetry else None
        if PerovskiteValidator.is_valid_halide_perovskite(
                doc.formula_pretty, comp_dict, sg):
            feats = extractor.extract_all_features(doc.composition, doc.formula_pretty)
            data.append({
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "e_above_hull": doc.energy_above_hull,
            })
            feature_list.append(feats)

    df_raw  = pd.DataFrame(data)
    feat_df = pd.DataFrame(feature_list).fillna(0)
    df_full = pd.concat([df_raw, feat_df], axis=1)
    os.makedirs(outputs_dir, exist_ok=True)
    df_full.to_csv(cache_path, index=False)
    print(f"  ✓ Saved feature cache: {cache_path} ({len(df_full)} compounds)")

    feat_cols = feat_df.columns.tolist()
    X = feat_df.values.astype(float)
    y = df_raw["band_gap"].values.astype(float)
    formulas = df_raw["formula"].tolist()
    groups = _get_groups(formulas)
    print(f"Featurized {len(y)} compounds, {X.shape[1]} features.")
    return X, y, groups


def _get_groups(formulas):
    """Extract B-site element group label for GroupShuffleSplit."""
    from pymatgen.core import Composition
    B_SITE = {
        "Bi","Sb","Cu","Ag","In","Ga","Sn","Ge","Fe","Mn","Co","Ni",
        "Ti","V","Cr","Zr","Nb","Pd","Pt","Au","Zn","Cd","Hg","Tl","Y","La"
    }
    groups = []
    for f in formulas:
        try:
            comp = Composition(f)
            b_elems = [e.symbol for e in comp.elements if e.symbol in B_SITE]
            groups.append(b_elems[0] if b_elems else "Other")
        except Exception:
            groups.append("Other")
    return np.array(groups)


def main():
    import os

    print("=" * 70)
    print("GP ACTIVE LEARNING BASELINE")
    print("=" * 70)
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    X, y, groups = load_data()

    n0, K, n_seeds = 200, 50, 10
    seeds = list(range(n_seeds))

    # ── Strategies to compare ─────────────────────────────────────────────
    strategies = [
        # (model_class, acquisition_fn, display_name)
        (EnsembleModel, random_acquisition,       "Ensemble + Random"),
        (EnsembleModel, uncertainty_acquisition,  "Ensemble + Uncertainty"),
        (GPModel,       random_acquisition,       "GP + Random"),
        (GPModel,       ucb_acquisition,          "GP + UCB (κ=2)"),
        (GPModel,       ei_acquisition,           "GP + EI"),
    ]

    all_results = []

    for model_cls, acq_fn, name in strategies:
        print(f"\nRunning: {name}")
        for seed in seeds:
            print(f"  seed {seed}...", end=" ", flush=True)
            try:
                res = run_trial(X, y, groups, model_cls, acq_fn, name,
                                n0=n0, K=K, seed=seed)
                all_results.append(res)
                print(f"ΔMAE = {res['delta_mae_pct']:+.2f}%")
            except Exception as e:
                print(f"ERROR: {e}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    df_res = pd.DataFrame(all_results)
    summary = (
        df_res.groupby("acquisition")["delta_mae_pct"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_delta_mae_pct",
                         "std":  "std_delta_mae_pct",
                         "count": "n_seeds"})
        .reset_index()
    )
    summary["se"] = summary["std_delta_mae_pct"] / np.sqrt(summary["n_seeds"])

    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY  (positive = MAE reduced after querying)")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"  {row['acquisition']:<35}  "
              f"ΔMAE = {row['mean_delta_mae_pct']:+.2f}% ± {row['std_delta_mae_pct']:.2f}%")

    # ── Statistical tests vs Ensemble+Random ─────────────────────────────
    print("\nPaired t-tests vs Ensemble + Random baseline:")
    baseline_vals = df_res[df_res["acquisition"] == "Ensemble + Random"]["delta_mae_pct"].values
    for name in df_res["acquisition"].unique():
        if name == "Ensemble + Random":
            continue
        other_vals = df_res[df_res["acquisition"] == name]["delta_mae_pct"].values
        if len(other_vals) == len(baseline_vals):
            _, p = ttest_rel(other_vals, baseline_vals)
            mean_diff = other_vals.mean() - baseline_vals.mean()
            print(f"  {name:<35}  diff = {mean_diff:+.2f}%  p = {p:.3f}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    df_res.to_csv("outputs/gp_al_comparison.csv", index=False)
    summary.to_csv("outputs/gp_al_summary.csv", index=False)
    print("\n✓ Saved: outputs/gp_al_comparison.csv")
    print("✓ Saved: outputs/gp_al_summary.csv")

    # ── Plot ──────────────────────────────────────────────────────────────
    strategy_order = [s[2] for s in strategies]
    colors = ["#AAAAAA", "#888888", "#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(strategy_order))
    means = [summary.loc[summary["acquisition"] == n, "mean_delta_mae_pct"].values[0]
             for n in strategy_order]
    ses   = [summary.loc[summary["acquisition"] == n, "se"].values[0]
             for n in strategy_order]

    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8,
                  yerr=ses, capsize=5, error_kw=dict(elinewidth=1.2))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_order, rotation=20, ha="right", fontsize=14)
    ax.set_ylabel("MAE reduction after 1 round (%)", fontsize=18)
    ax.set_title("GP vs Ensemble Active Learning Baseline\n"
                 f"(n₀={n0}, K={K}, {n_seeds} seeds, group-based split)",
                 fontsize=18)
    ax.tick_params(labelsize=14)

    # Significance annotations
    baseline_mean = summary.loc[summary["acquisition"] == "Ensemble + Random",
                                "mean_delta_mae_pct"].values[0]
    for i, name in enumerate(strategy_order):
        if name == "Ensemble + Random":
            continue
        other_vals = df_res[df_res["acquisition"] == name]["delta_mae_pct"].values
        if len(other_vals) == len(baseline_vals):
            _, p = ttest_rel(other_vals, baseline_vals)
            label = "ns" if p >= 0.05 else ("*" if p >= 0.01 else "**")
            ax.text(i, means[i] + ses[i] + 0.1, label,
                    ha="center", va="bottom", fontsize=14)

    plt.tight_layout()
    fig.savefig("outputs/gp_al_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: outputs/gp_al_comparison.png")
    plt.close()

    return df_res, summary


if __name__ == "__main__":
    df_res, summary = main()
