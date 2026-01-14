# Lead-Free Halide Perovskite Screening with Active Learning

## Overview

This repository contains an end-to-end, uncertainty-aware active learning pipeline for discovering stable, lead-free halide perovskite photovoltaic materials.
The workflow retrieves data from the Materials Project, engineers chemistry-informed features, trains an ensemble model with calibrated uncertainties, and recommends candidates for validation.

## What This Pipeline Does

- **Data acquisition** from the Materials Project (lead-free, halide, perovskite-like compositions).
- **Feature engineering** with 35+ compositional and perovskite descriptors.
- **Ensemble modeling** with uncertainty estimation.
- **PBE-to-experimental calibration** using stratified regression with bootstrap confidence.
- **Physics-based filtering** to remove implausible candidates.
- **Active learning prioritization** to suggest the next materials to validate.
- **Advanced analyses** for uncertainty calibration, generalization testing, and active-learning simulations.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you prefer explicit installs:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
pip install mp-api pymatgen pytest pyyaml python-dotenv
```

## Environment Setup

Create a `.env` file and set your Materials Project API key:

```bash
MATERIALS_PROJECT_API_KEY=your_api_key_here
```

Get a key at: https://next-gen.materialsproject.org/api

## Usage Flow

### 1) Run the Full Pipeline

```bash
python improved_perovskite_screening.py
```

### 2) Review Outputs

The pipeline writes CSV results and figures into `./outputs` (see [Expected Outputs](#expected-outputs)).

### 3) Run Tests

```bash
pytest test_improved_perovskite.py -v
```

## Scripts Executed by the Main Pipeline

`improved_perovskite_screening.py` dynamically imports and runs the following analysis modules when those steps are enabled:

1. **`uncertainty_calibration_analysis.py`**
   - Runs uncertainty calibration diagnostics (correlation, ECE, coverage, NLL, CRPS).
2. **`generalization_analysis.py`**
   - Evaluates generalization via random, group, LOEO, and cluster-based splits.
3. **`active_learning_simulation.py`**
   - Retrospective active-learning simulation against random baselines.
4. **`calibration_sensitivity.py`**
   - Sensitivity study for PBE calibration (LOOCV + bootstrap).
5. **`proxy_features_ablation.py`**
   - Proxy-feature ablation experiments.
6. **`active_learning_multiround.py`**
   - Multi-round active-learning learning curves.

> If you only want the core screening flow, you can disable steps 3.5/3.6/6.5/6.6/6.7/6.8 inside `improved_perovskite_screening.py`.

## Pipeline Workflow (High-Level)

```
Step 1: Data acquisition (Materials Project)
Step 2: Feature engineering (35+ descriptors)
Step 3: Ensemble training + uncertainty
Step 3.5: Uncertainty calibration analysis (optional)
Step 3.6: Generalization analysis (optional)
Step 4: PBE-to-experiment calibration
Step 5: Screening & physics-based filtering
Step 6: Active learning recommendations
Step 6.5: AL simulation (optional)
Step 6.6: Calibration sensitivity (optional)
Step 6.7: Proxy-feature ablation (optional)
Step 6.8: Multi-round AL (optional)
Step 7: Visualization & reporting
```

## Expected Outputs

```
./outputs/
├── improved_candidates.csv                         # Top screening candidates
├── active_learning_recommendations_improved.csv    # DFT validation priorities
├── improved_model_performance.png                  # Parity plot + residuals
├── improved_pbe_calibration.png                    # PBE→Exp calibration
├── improved_uncertainty_analysis.png               # Uncertainty landscape
├── improved_feature_importance.png                 # Feature rankings
│
├── uncertainty_calibration_diagnostics.png         # 4-panel calibration plot
├── uncertainty_calibration_summary.csv             # Calibration metrics
├── generalization_comparison.png                   # Generalization plots
├── generalization_summary.csv                      # Performance across splits
├── active_learning_results.png                     # Learning curves
└── active_learning_summary.csv                     # AL improvement stats
```

## Configuration

Edit `Config` in `improved_perovskite_screening.py` to adjust thresholds and model sizes:

```python
@dataclass
class Config:
    target_bandgap: float = 1.34
    bandgap_tolerance: float = 0.25
    max_e_above_hull: float = 0.05

    max_uncertainty: float = 0.5
    min_uncertainty: float = 0.05

    n_ensemble_models: int = 15
    active_learning_batch_size: int = 15
```

## Testing Notes

- `test_improved_perovskite.py` contains unit and integration tests.
- Tests are **assertion-based** and **do not write CSVs or plots**; they only report results via pytest output.

## License

MIT License.
