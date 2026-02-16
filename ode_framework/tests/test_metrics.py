"""Tests for error metrics.

Unit tests for error metric functions using pytest.
"""
import pytest
import numpy as np

from ode_framework.metrics.error_metrics import (
    l2_norm,
    mse,
    rmse,
    r_squared,
    compute_all_metrics,
)


class TestPerfectPrediction:
    """Tests for perfect predictions (y_true == y_pred)."""

    def test_perfect_prediction_1d(self):
        """Test all metrics on 1D perfect prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        assert l2_norm(y_true, y_pred) == 0.0
        assert mse(y_true, y_pred) == 0.0
        assert rmse(y_true, y_pred) == 0.0
        assert r_squared(y_true, y_pred) == 1.0

    def test_perfect_prediction_2d(self):
        """Test all metrics on 2D perfect prediction."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = y_true.copy()

        assert l2_norm(y_true, y_pred) == 0.0
        assert mse(y_true, y_pred) == 0.0
        assert rmse(y_true, y_pred) == 0.0
        assert r_squared(y_true, y_pred) == 1.0

    def test_perfect_prediction_via_compute_all_metrics(self):
        """Test compute_all_metrics on perfect prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()

        metrics = compute_all_metrics(y_true, y_pred)

        assert metrics["l2_norm"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r_squared"] == 1.0


class TestL2NormCalculation:
    """Tests for L2 norm metric."""

    def test_l2_norm_simple_1d(self):
        """Test L2 norm with simple 1D array."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = l2_norm(y_true, y_pred)
        assert result == 0.0

    def test_l2_norm_known_calculation(self):
        """Test L2 norm with hand-calculated result.

        Error = [0.1, 0.2, 0.3]
        L2 = sqrt(0.1^2 + 0.2^2 + 0.3^2) = sqrt(0.14) ≈ 0.374166
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.9, 1.8, 2.7])

        result = l2_norm(y_true, y_pred)
        expected = np.sqrt(0.01 + 0.04 + 0.09)

        assert np.isclose(result, expected)

    def test_l2_norm_negative_errors(self):
        """Test L2 norm with negative errors."""
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 0.0, 1.5])

        result = l2_norm(y_true, y_pred)
        expected = np.sqrt(1.0 + 1.0 + 0.25)

        assert np.isclose(result, expected)

    def test_l2_norm_2d_array(self):
        """Test L2 norm with 2D array."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.1], [2.9, 3.9]])

        error = y_true - y_pred
        expected = np.sqrt(np.sum(error**2))
        result = l2_norm(y_true, y_pred)

        assert np.isclose(result, expected)

    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
    def test_l2_norm_scaling(self, scale):
        """Test L2 norm scales correctly with data magnitude."""
        y_true = np.array([1.0, 2.0, 3.0]) * scale
        y_pred = np.array([1.1, 2.1, 2.9]) * scale

        error = y_true - y_pred
        expected = np.sqrt(np.sum(error**2))
        result = l2_norm(y_true, y_pred)

        assert np.isclose(result, expected)


class TestMSECalculation:
    """Tests for Mean Squared Error metric."""

    def test_mse_simple(self):
        """Test MSE with simple calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        expected = np.mean(np.array([0.01, 0.01, 0.01]))
        result = mse(y_true, y_pred)

        assert np.isclose(result, expected)

    def test_mse_known_values(self):
        """Test MSE with hand-calculated values.

        Errors: [0.5, -0.5, 1.0]
        MSE = (0.25 + 0.25 + 1.0) / 3 = 1.5 / 3 = 0.5
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.5, 2.5, 2.0])

        result = mse(y_true, y_pred)
        expected = 0.5

        assert np.isclose(result, expected)

    def test_mse_2d_array(self):
        """Test MSE on 2D array."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.2], [2.9, 3.8], [5.1, 5.9]])

        error = y_true - y_pred
        expected = np.mean(error**2)
        result = mse(y_true, y_pred)

        assert np.isclose(result, expected)

    def test_mse_always_positive(self):
        """Test that MSE is always non-negative."""
        y_true = np.array([1.0, -2.0, 3.0, -4.0])
        y_pred = np.array([1.5, -2.5, 2.5, -3.5])

        result = mse(y_true, y_pred)
        assert result >= 0.0

    def test_mse_vs_rmse_relationship(self):
        """Test that RMSE = sqrt(MSE)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        mse_val = mse(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)

        assert np.isclose(rmse_val, np.sqrt(mse_val))


class TestRMSECalculation:
    """Tests for Root Mean Squared Error metric."""

    def test_rmse_simple(self):
        """Test RMSE with simple calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        result = rmse(y_true, y_pred)
        assert result == 0.0

    def test_rmse_known_values(self):
        """Test RMSE with hand-calculated values.

        Errors: [0.1, 0.1, 0.1, 0.1]
        MSE = 0.01
        RMSE = 0.1
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])

        result = rmse(y_true, y_pred)
        expected = 0.1

        assert np.isclose(result, expected)

    def test_rmse_2d_array(self):
        """Test RMSE on 2D array."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.2], [2.9, 3.8]])

        error = y_true - y_pred
        expected = np.sqrt(np.mean(error**2))
        result = rmse(y_true, y_pred)

        assert np.isclose(result, expected)

    def test_rmse_same_units_as_data(self):
        """Test that RMSE is in same units as prediction."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 31.0])

        result = rmse(y_true, y_pred)
        assert np.isclose(result, 1.0)


class TestRSquaredCalculation:
    """Tests for R² coefficient of determination."""

    def test_r_squared_perfect_fit(self):
        """Test R² = 1 for perfect fit."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        result = r_squared(y_true, y_pred)
        assert result == 1.0

    def test_r_squared_mean_baseline(self):
        """Test R² = 0 when predictions equal mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))

        result = r_squared(y_true, y_pred)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_r_squared_worse_than_mean(self):
        """Test R² < 0 for predictions worse than mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        result = r_squared(y_true, y_pred)
        assert result < 0.0

    def test_r_squared_linear_data(self):
        """Test R² on linear data with perfect linear fit."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_true = 2 * x + 1

        y_pred = y_true.copy()

        result = r_squared(y_true, y_pred)
        assert np.isclose(result, 1.0)

    def test_r_squared_known_calculation(self):
        """Test R² with hand-calculated values.

        y_true = [1, 2, 3]
        y_pred = [1, 2, 3] (perfect)
        SS_res = 0, SS_tot = 2, R² = 1 - 0/2 = 1
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = r_squared(y_true, y_pred)
        assert result == 1.0

    def test_r_squared_constant_data_perfect_pred(self):
        """Test R² on constant y_true with perfect prediction."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])

        result = r_squared(y_true, y_pred)
        assert result == 1.0

    def test_r_squared_constant_data_imperfect_pred(self):
        """Test R² on constant y_true with imperfect prediction."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.1, 5.1, 5.1, 5.1])

        result = r_squared(y_true, y_pred)
        assert result == 0.0

    def test_r_squared_2d_array(self):
        """Test R² on 2D array."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [2.9, 3.9], [5.1, 5.9]])

        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        expected = 1.0 - (ss_res / ss_tot)

        result = r_squared(y_true, y_pred)
        assert np.isclose(result, expected)

    @pytest.mark.parametrize("r2_range", [0.5, 0.7, 0.9, 0.99])
    def test_r_squared_range(self, r2_range):
        """Test R² values in typical range."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        noise = np.sqrt((1 - r2_range) / r2_range) * np.std(y_true)
        y_pred = y_true + np.random.RandomState(42).normal(0, noise, len(y_true))

        result = r_squared(y_true, y_pred)
        assert 0.0 <= result <= 1.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_compute_all_metrics_returns_dict(self):
        """Test that compute_all_metrics returns dictionary with correct keys."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        result = compute_all_metrics(y_true, y_pred)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"l2_norm", "mse", "rmse", "r_squared"}

    def test_compute_all_metrics_perfect_prediction(self):
        """Test compute_all_metrics on perfect prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()

        metrics = compute_all_metrics(y_true, y_pred)

        assert metrics["l2_norm"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r_squared"] == 1.0

    def test_compute_all_metrics_consistency(self):
        """Test that compute_all_metrics matches individual function calls."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])

        metrics = compute_all_metrics(y_true, y_pred)

        assert np.isclose(metrics["l2_norm"], l2_norm(y_true, y_pred))
        assert np.isclose(metrics["mse"], mse(y_true, y_pred))
        assert np.isclose(metrics["rmse"], rmse(y_true, y_pred))
        assert np.isclose(metrics["r_squared"], r_squared(y_true, y_pred))

    def test_compute_all_metrics_2d_array(self):
        """Test compute_all_metrics on 2D array."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [2.9, 3.9], [5.1, 5.9]])

        metrics = compute_all_metrics(y_true, y_pred)

        assert all(isinstance(v, float) for v in metrics.values())
        assert all(np.isfinite(v) for v in metrics.values())


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="same shape"):
            l2_norm(y_true, y_pred)

    def test_nan_in_y_true_raises_error(self):
        """Test that NaN in y_true raises ValueError."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="NaN"):
            l2_norm(y_true, y_pred)

    def test_nan_in_y_pred_raises_error(self):
        """Test that NaN in y_pred raises ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="NaN"):
            mse(y_true, y_pred)

    def test_inf_in_y_true_raises_error(self):
        """Test that Inf in y_true raises ValueError."""
        y_true = np.array([1.0, np.inf, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Inf"):
            rmse(y_true, y_pred)

    def test_inf_in_y_pred_raises_error(self):
        """Test that Inf in y_pred raises ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, -np.inf, 3.0])

        with pytest.raises(ValueError, match="Inf"):
            r_squared(y_true, y_pred)

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError, match="empty"):
            l2_norm(y_true, y_pred)

    def test_non_array_input_raises_error(self):
        """Test that non-array inputs raise TypeError."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = np.array([1.0, 2.0, 3.0])

        with pytest.raises(TypeError):
            l2_norm(y_true, y_pred)

    @pytest.mark.parametrize("metric_func", [l2_norm, mse, rmse, r_squared])
    def test_all_metrics_reject_nan(self, metric_func):
        """Test that all metric functions reject NaN values."""
        y_true = np.array([1.0, 2.0, np.nan])
        y_pred = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            metric_func(y_true, y_pred)

    @pytest.mark.parametrize("metric_func", [l2_norm, mse, rmse, r_squared])
    def test_all_metrics_reject_mismatched_shapes(self, metric_func):
        """Test that all metric functions reject mismatched shapes."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError):
            metric_func(y_true, y_pred)


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_very_small_errors(self):
        """Test metrics with very small errors."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true + 1e-15

        l2 = l2_norm(y_true, y_pred)
        assert l2 >= 0.0
        assert np.isfinite(l2)

    def test_very_large_values(self):
        """Test metrics with very large values."""
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = y_true + 1e9

        mse_val = mse(y_true, y_pred)
        assert np.isfinite(mse_val)
        assert mse_val > 0.0

    def test_mixed_scale_values(self):
        """Test metrics with mixed scale values."""
        y_true = np.array([1e-10, 1.0, 1e10])
        y_pred = np.array([1.1e-10, 1.1, 1.1e10])

        metrics = compute_all_metrics(y_true, y_pred)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_repeated_values(self):
        """Test metrics when y_true has repeated values."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        result = r_squared(y_true, y_pred)
        assert result == 1.0

    def test_negative_values(self):
        """Test metrics with negative values."""
        y_true = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        y_pred = np.array([-4.9, -2.9, -1.1, 1.1, 3.1])

        l2 = l2_norm(y_true, y_pred)
        assert l2 > 0.0
        assert np.isfinite(l2)

    @pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
    def test_metrics_scale_with_sample_size(self, n_samples):
        """Test that MSE and RMSE scale appropriately with sample size."""
        y_true = np.random.RandomState(42).normal(0, 1, n_samples)
        error = np.random.RandomState(43).normal(0, 0.1, n_samples)
        y_pred = y_true + error

        mse_val = mse(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)

        assert np.isfinite(mse_val)
        assert np.isfinite(rmse_val)
        assert rmse_val == np.sqrt(mse_val)
