import pytest
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from improved_perovskite_screening import (
    EnhancedFeatureExtractor,
    PerovskiteValidator,
    StratifiedPBECalibrator,
    ImprovedEnsemblePredictor,
    Config
)


class TestConfig:
    def test_config_defaults(self):
        config = Config()

        assert config.target_bandgap == 1.34
        assert config.n_ensemble_models >= 10
        assert config.max_e_above_hull == 0.05
        assert 0 < config.test_size < 1


class TestEnhancedFeatureExtractor:
    def test_compositional_features_simple(self):
        comp = Composition("CsSnI3")
        features = EnhancedFeatureExtractor.get_compositional_features(comp)

        expected_features = [
            "Z_mean", "Z_std", "Z_range",
            "EN_mean", "EN_std", "EN_range",
            "Radius_mean", "Radius_std",
            "N_elements", "metal_ratio"
        ]

        for feat in expected_features:
            assert feat in features, f"Missing feature: {feat}"

        assert features["N_elements"] == 3
        assert 0 <= features["metal_ratio"] <= 1
        assert features["EN_mean"] > 0

    def test_perovskite_descriptors(self):
        comp = Composition("CsSnI3")
        descriptors = EnhancedFeatureExtractor.get_perovskite_descriptors(
            "CsSnI3", comp
        )

        assert descriptors["A_site_count"] == 1
        assert descriptors["B_site_count"] == 1
        assert descriptors["X_site_count"] == 3

        assert abs(descriptors["A_to_B_ratio"] - 1.0) < 0.01
        assert abs(descriptors["X_to_B_ratio"] - 3.0) < 0.01

    def test_double_perovskite_descriptors(self):
        comp = Composition("Cs2AgBiBr6")
        descriptors = EnhancedFeatureExtractor.get_perovskite_descriptors(
            "Cs2AgBiBr6", comp
        )

        assert abs(descriptors["A_to_B_ratio"] - 1.0) < 0.01
        assert abs(descriptors["X_to_B_ratio"] - 3.0) < 0.01

    def test_extract_all_features(self):
        comp = Composition("CsSnI3")
        features = EnhancedFeatureExtractor.extract_all_features(comp, "CsSnI3")

        assert isinstance(features, pd.Series)
        assert len(features) >= 25
        assert not features.isna().any()


class TestPerovskiteValidator:
    def test_valid_simple_perovskite(self):
        comp_dict = {"Cs": 1, "Sn": 1, "I": 3}

        is_valid = PerovskiteValidator.is_valid_halide_perovskite(
            "CsSnI3", comp_dict
        )

        assert is_valid, "CsSnI3 should be valid"

    def test_valid_double_perovskite(self):
        comp_dict = {"Cs": 2, "Ag": 1, "Bi": 1, "Br": 6}

        is_valid = PerovskiteValidator.is_valid_halide_perovskite(
            "Cs2AgBiBr6", comp_dict
        )

        assert is_valid, "Cs2AgBiBr6 should be valid"

    def test_invalid_no_halide(self):
        comp_dict = {"Cs": 2, "Ti": 1, "O": 3}

        is_valid = PerovskiteValidator.is_valid_halide_perovskite(
            "Cs2TiO3", comp_dict
        )

        assert not is_valid, "Cs2TiO3 (oxide) should be invalid"

    def test_invalid_stoichiometry(self):
        comp_dict = {"Cs": 1, "Sn": 1, "I": 2}

        is_valid = PerovskiteValidator.is_valid_halide_perovskite(
            "CsSnI2", comp_dict
        )

        assert not is_valid, "CsSnI2 has incorrect stoichiometry"

    def test_invalid_too_many_elements(self):
        comp_dict = {"Cs": 1, "Rb": 1, "Sn": 1, "Ge": 1, "I": 3, "Br": 3}

        is_valid = PerovskiteValidator.is_valid_halide_perovskite(
            "CsRbSnGeI3Br3", comp_dict
        )

        assert not is_valid, "Too many elements (6) should be invalid"

    def test_physical_reasonableness_licuf(self):
        is_valid, reason = PerovskiteValidator.is_physically_reasonable(
            "LiCuF3", predicted_gap=1.34, uncertainty=0.15
        )

        assert not is_valid, "LiCuF3 should be flagged as unreasonable"
        assert "wide-gap" in reason.lower()

    def test_physical_reasonableness_high_uncertainty(self):
        is_valid, reason = PerovskiteValidator.is_physically_reasonable(
            "CsSnI3", predicted_gap=1.34, uncertainty=0.8
        )

        assert not is_valid, "High uncertainty should be flagged"
        assert "uncertainty" in reason.lower()

    def test_physical_reasonableness_polyatomic_anion(self):
        is_valid, reason = PerovskiteValidator.is_physically_reasonable(
            "Sr2CuSe2(ClO3)2", predicted_gap=1.5, uncertainty=0.2
        )

        assert not is_valid, "ClO3 (polyatomic) should be rejected"
        assert "polyatomic" in reason.lower()


class TestStratifiedPBECalibrator:
    def test_calibration_fitting(self):
        calibrator = StratifiedPBECalibrator()
        calibrator.fit(n_bootstrap=100)

        assert calibrator.global_model is not None
        assert "slope" in calibrator.global_model
        assert "intercept" in calibrator.global_model
        assert "r2" in calibrator.global_model

        assert 0.5 < calibrator.global_model["slope"] < 1.5
        assert 0 < calibrator.global_model["intercept"] < 2
        assert 0.5 < calibrator.global_model["r2"] < 1.0

    def test_calibration_values(self):
        calibrator = StratifiedPBECalibrator()
        calibrator.fit(n_bootstrap=100)

        pbe_gap = 0.86
        exp_gap_predicted = calibrator.calibrate(np.array([pbe_gap]))[0]

        assert abs(exp_gap_predicted - 1.27) < abs(pbe_gap - 1.27)
        assert 1.0 < exp_gap_predicted < 1.5

    def test_bootstrap_confidence(self):
        calibrator = StratifiedPBECalibrator()
        calibrator.fit(n_bootstrap=100)

        assert len(calibrator.bootstrap_slopes) == 100
        assert len(calibrator.bootstrap_intercepts) == 100

        slope_range = max(calibrator.bootstrap_slopes) - min(calibrator.bootstrap_slopes)
        ci_width = np.percentile(calibrator.bootstrap_slopes, 97.5) - np.percentile(
            calibrator.bootstrap_slopes, 2.5
        )

        assert ci_width < slope_range


class TestImprovedEnsemblePredictor:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1

        return X, y

    def test_model_initialization(self):
        model = ImprovedEnsemblePredictor(n_models=5)

        assert model.n_models == 5
        assert len(model.models) == 0

    def test_model_fitting(self, synthetic_data):
        X, y = synthetic_data

        model = ImprovedEnsemblePredictor(n_models=5)
        model.fit(X, y)

        assert len(model.models) == 5
        assert model.scaler is not None

    def test_prediction_with_uncertainty(self, synthetic_data):
        X, y = synthetic_data
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]

        model = ImprovedEnsemblePredictor(n_models=5)
        model.fit(X_train, y_train)

        mean_pred, std_pred, all_preds = model.predict_with_uncertainty(X_test)

        assert mean_pred.shape == (20,)
        assert std_pred.shape == (20,)
        assert all_preds.shape == (5, 20)

        assert np.all(std_pred >= 0)
        assert np.allclose(mean_pred, all_preds.mean(axis=0))

    def test_feature_importance(self, synthetic_data):
        X, y = synthetic_data

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        model = ImprovedEnsemblePredictor(n_models=3)
        model.fit(X, y, feature_names=feature_names)

        importance_df = model.get_feature_importance()

        assert importance_df is not None
        assert len(importance_df) == X.shape[1]
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns


class TestIntegration:
    def test_end_to_end_simple(self):
        materials = [
            {"formula": "CsSnI3", "comp": Composition("CsSnI3"), "gap": 0.86},
            {"formula": "CsSnBr3", "comp": Composition("CsSnBr3"), "gap": 1.47},
            {"formula": "Cs2AgBiBr6", "comp": Composition("Cs2AgBiBr6"), "gap": 1.89},
            {"formula": "CsGeI3", "comp": Composition("CsGeI3"), "gap": 0.95},
        ]

        extractor = EnhancedFeatureExtractor()
        feature_list = []
        gaps = []

        for mat in materials:
            features = extractor.extract_all_features(mat["comp"], mat["formula"])
            feature_list.append(features)
            gaps.append(mat["gap"])

        feature_df = pd.DataFrame(feature_list).fillna(0)
        X = feature_df.values
        y = np.array(gaps)

        model = ImprovedEnsemblePredictor(n_models=3)
        model.fit(X, y)

        pred, uncert, _ = model.predict_with_uncertainty(X)

        assert len(pred) == len(materials)
        assert len(uncert) == len(materials)
        assert np.all(uncert >= 0)

        assert np.mean(np.abs(pred - y)) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
