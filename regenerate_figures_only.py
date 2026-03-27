#!/usr/bin/env python3
"""
Regenerate the 4 main figures only: model performance, calibration,
uncertainty analysis, feature importance.
Skips AL simulation / generalization / proxy ablation.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from improved_perovskite_screening import (
    Config, EnhancedFeatureExtractor, PerovskiteValidator,
    ImprovedEnsemblePredictor, StratifiedPBECalibrator, ImprovedVisualizer
)
from mp_api.client import MPRester

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# ── Step 1: Data ──
print("Step 1: Fetching data from Materials Project...")
data = []
with MPRester(config.api_key) as mpr:
    docs = mpr.materials.summary.search(
        exclude_elements=["Pb"],
        num_elements=(config.min_elements, config.max_elements),
        band_gap=(config.bandgap_min, config.bandgap_max),
        fields=["material_id","formula_pretty","band_gap",
                "energy_above_hull","composition","elements","symmetry"]
    )

print(f"  API returned {len(docs)} materials")
for doc in docs:
    comp_dict = doc.composition.as_dict()
    sg = doc.symmetry.number if hasattr(doc,'symmetry') and doc.symmetry else None
    if PerovskiteValidator.is_valid_halide_perovskite(doc.formula_pretty, comp_dict, sg):
        data.append({
            'material_id': doc.material_id,
            'formula': doc.formula_pretty,
            'band_gap': doc.band_gap,
            'e_above_hull': doc.energy_above_hull,
            'composition_obj': doc.composition,
            'space_group': sg
        })
df = pd.DataFrame(data)
print(f"  Filtered to {len(df)} compounds")

# ── Step 2: Features ──
print("Step 2: Feature engineering...")
ext = EnhancedFeatureExtractor()
feature_list = [ext.extract_all_features(row['composition_obj'], row['formula'])
                for _, row in df.iterrows()]
feature_df = pd.DataFrame(feature_list).fillna(0)
X = feature_df.values
y = df['band_gap'].values
print(f"  {X.shape[1]} features")

# ── Step 3: Train / Test ──
print("Step 3: Training ensemble on train split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.test_size, random_state=config.random_state)

model = ImprovedEnsemblePredictor(n_models=config.n_ensemble_models,
                                   random_state=config.random_state)
model.fit(X_train, y_train, feature_names=feature_df.columns.tolist())
y_pred, y_std, _ = model.predict_with_uncertainty(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  R²={r2:.3f}  RMSE={rmse:.3f} eV")

# ── Calibration ──
print("Step 3b: Calibration...")
calibrator = StratifiedPBECalibrator()
calibrator.fit()
a = calibrator.global_model['slope']
b = calibrator.global_model['intercept']

# Full-dataset model for uncertainty analysis + feature importance
print("Step 3c: Retraining on full dataset for uncertainty analysis...")
model_full = ImprovedEnsemblePredictor(n_models=config.n_ensemble_models,
                                        random_state=config.random_state)
model_full.fit(X, y, feature_names=feature_df.columns.tolist())
y_pred_full, y_std_full, _ = model_full.predict_with_uncertainty(X)
df['predicted_gap_exp'] = a * y_pred_full + b
df['prediction_uncertainty'] = y_std_full

# ── Step 7: Figures ──
print("Step 7: Generating figures...")
viz = ImprovedVisualizer()

viz.plot_model_performance(
    y_test, y_pred, y_std,
    os.path.join(config.save_dir, 'improved_model_performance.png'))
print("  ✓ improved_model_performance.png")

calibrator.plot_calibration(
    os.path.join(config.save_dir, 'improved_pbe_calibration.png'))
print("  ✓ improved_pbe_calibration.png")

viz.plot_uncertainty_analysis(
    df['predicted_gap_exp'].values,
    df['prediction_uncertainty'].values,
    df['e_above_hull'].values,
    os.path.join(config.save_dir, 'improved_uncertainty_analysis.png'),
    target_gap=config.target_bandgap,
    bandgap_tolerance=config.bandgap_tolerance,
    min_uncertainty=config.min_uncertainty,
    max_uncertainty=config.max_uncertainty)
print("  ✓ improved_uncertainty_analysis.png")

importance_df = model_full.get_feature_importance()
if importance_df is not None:
    viz.plot_feature_importance(
        importance_df,
        os.path.join(config.save_dir, 'improved_feature_importance.png'),
        top_n=15)
    print("  ✓ improved_feature_importance.png")

print("\nDone! All 4 main figures regenerated.")
