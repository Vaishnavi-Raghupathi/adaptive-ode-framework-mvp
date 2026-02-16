"""Tests for ODE solvers.

Unit tests for RK4Solver and RK45Solver implementations using pytest.
"""
import pytest
import numpy as np

from ode_framework.solvers.classical import RK4Solver, RK45Solver
from ode_framework.utils.test_problems import (
    exponential_decay,
    harmonic_oscillator,
    logistic_growth,
)
from ode_framework.metrics.error_metrics import rmse, compute_all_metrics


# Fixtures for test data generation
@pytest.fixture
def exponential_clean_data():
    """Clean exponential decay data without noise."""
    t = np.linspace(0, 5, 50)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5)
    return t, problem["x_exact"], problem["params"]


@pytest.fixture
def exponential_noisy_data():
    """Exponential decay data with 5% noise."""
    t = np.linspace(0, 5, 50)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5, noise_level=0.05)
    return t, problem["x_exact"], problem["params"]


@pytest.fixture
def harmonic_clean_data():
    """Clean harmonic oscillator data."""
    t = np.linspace(0, 4 * np.pi, 100)
    problem = harmonic_oscillator(t, x0=1.0, v0=0.0, omega=1.0)
    return t, problem["x_exact"], problem["params"]


@pytest.fixture
def logistic_clean_data():
    """Clean logistic growth data."""
    t = np.linspace(0, 10, 100)
    problem = logistic_growth(t, x0=0.1, r=1.0, K=1.0)
    return t, problem["x_exact"], problem["params"]


@pytest.fixture(params=[RK4Solver, RK45Solver])
def solver_class(request):
    """Parametrized fixture for both solver classes."""
    return request.param


class TestExponentialDecay:
    """Tests for exponential decay problem."""

    def test_exponential_decay_rk4(self, exponential_clean_data):
        """Test RK4Solver on exponential decay problem."""
        t, x_exact, params = exponential_clean_data

        solver = RK4Solver()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.01, f"RMSE {rmse_val} exceeds threshold 0.01"

    def test_exponential_decay_rk45(self, exponential_clean_data):
        """Test RK45Solver on exponential decay problem."""
        t, x_exact, params = exponential_clean_data

        solver = RK45Solver()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.01, f"RMSE {rmse_val} exceeds threshold 0.01"

    @pytest.mark.parametrize("solver_class_type", [RK4Solver, RK45Solver])
    def test_exponential_decay_both_solvers(
        self, solver_class_type, exponential_clean_data
    ):
        """Test both solvers on exponential decay."""
        t, x_exact, params = exponential_clean_data

        solver = solver_class_type()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        metrics = compute_all_metrics(x_exact, x_pred)
        assert metrics["rmse"] < 0.01
        assert metrics["r_squared"] > 0.99

    def test_exponential_with_noisy_data(self, exponential_noisy_data):
        """Test solvers with 5% noise on exponential decay."""
        t, x_noisy, params = exponential_noisy_data

        for solver_class in [RK4Solver, RK45Solver]:
            solver = solver_class()
            solver.fit(t, x_noisy)
            x_pred = solver.predict(t)

            rmse_val = rmse(x_noisy, x_pred)
            assert rmse_val < 0.1, (
                f"{solver_class.__name__}: "
                f"RMSE {rmse_val} exceeds threshold 0.1 with noise"
            )

    def test_residual_computation_exponential(self, exponential_clean_data):
        """Test residual computation on exponential decay."""
        t, x_exact, params = exponential_clean_data

        for solver_class in [RK4Solver, RK45Solver]:
            solver = solver_class()
            solver.fit(t, x_exact)

            residuals = solver.compute_residuals(t, x_exact)

            assert residuals.shape == x_exact.shape, (
                f"Residual shape {residuals.shape} "
                f"doesn't match x_exact shape {x_exact.shape}"
            )
            assert np.abs(np.mean(residuals)) < 0.01, (
                f"Mean residual {np.mean(residuals)} should be close to zero"
            )
            assert np.all(np.isfinite(residuals)), "Residuals contain NaN/Inf"


class TestHarmonicOscillator:
    """Tests for harmonic oscillator problem."""

    def test_harmonic_oscillator_rk4(self, harmonic_clean_data):
        """Test RK4Solver on harmonic oscillator."""
        t, x_exact, params = harmonic_clean_data

        solver = RK4Solver()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.05, (
            f"RK4Solver RMSE {rmse_val} exceeds threshold 0.05 "
            f"for harmonic oscillator"
        )

    def test_harmonic_oscillator_rk45(self, harmonic_clean_data):
        """Test RK45Solver on harmonic oscillator."""
        t, x_exact, params = harmonic_clean_data

        solver = RK45Solver()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.05, (
            f"RK45Solver RMSE {rmse_val} exceeds threshold 0.05 "
            f"for harmonic oscillator"
        )

    def test_harmonic_energy_preservation(self, harmonic_clean_data):
        """Test that harmonic oscillator approximately preserves energy."""
        t, x_exact, params = harmonic_clean_data
        omega = params["omega"]

        for solver_class in [RK4Solver, RK45Solver]:
            solver = solver_class()
            solver.fit(t, x_exact)
            x_pred = solver.predict(t)

            x_vals = x_pred[:, 0]
            v_vals = x_pred[:, 1]

            energy = 0.5 * (v_vals**2 + omega**2 * x_vals**2)
            energy_variance = np.var(energy)

            assert energy_variance < 0.01, (
                f"{solver_class.__name__}: Energy variance {energy_variance} "
                f"indicates poor energy preservation"
            )


class TestLogisticGrowth:
    """Tests for logistic growth problem."""

    def test_logistic_growth_rk4(self, logistic_clean_data):
        """Test RK4Solver on logistic growth problem."""
        t, x_exact, params = logistic_clean_data

        solver = RK4Solver()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.05, (
            f"RK4Solver RMSE {rmse_val} exceeds threshold 0.05 "
            f"for logistic growth"
        )

    def test_logistic_growth_rk45(self, logistic_clean_data):
        """Test RK45Solver on logistic growth problem."""
        t, x_exact, params = logistic_clean_data

        solver = RK45Solver()
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.05, (
            f"RK45Solver RMSE {rmse_val} exceeds threshold 0.05 "
            f"for logistic growth"
        )

    def test_logistic_carrying_capacity(self, logistic_clean_data):
        """Test that logistic growth approaches carrying capacity."""
        t, x_exact, params = logistic_clean_data
        K = params["K"]

        for solver_class in [RK4Solver, RK45Solver]:
            solver = solver_class()
            solver.fit(t, x_exact)
            x_pred = solver.predict(t)

            final_value = x_pred[-1, 0]
            assert np.abs(final_value - K) < 0.1, (
                f"{solver_class.__name__}: Final value {final_value} "
                f"should approach carrying capacity K={K}"
            )


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises RuntimeError."""
        solver = RK4Solver()
        t_eval = np.linspace(0, 1, 10)

        with pytest.raises(RuntimeError, match="must be fitted"):
            solver.predict(t_eval)

    def test_residuals_before_fit_raises_error(self):
        """Test that compute_residuals before fit raises RuntimeError."""
        solver = RK4Solver()
        t = np.linspace(0, 1, 10)
        x = np.sin(t).reshape(-1, 1)

        with pytest.raises(RuntimeError, match="must be fitted"):
            solver.compute_residuals(t, x)

    def test_fit_with_nan_in_x_data(self):
        """Test that fit rejects data with NaN values."""
        solver = RK4Solver()
        t = np.linspace(0, 1, 10)
        x = np.sin(t).reshape(-1, 1)
        x[5, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            solver.fit(t, x)

    def test_fit_with_nan_in_t_data(self):
        """Test that fit rejects time data with NaN values."""
        solver = RK4Solver()
        t = np.linspace(0, 1, 10)
        x = np.sin(t).reshape(-1, 1)
        t[5] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            solver.fit(t, x)

    def test_fit_with_shape_mismatch(self):
        """Test that fit rejects mismatched shapes."""
        solver = RK4Solver()
        t = np.linspace(0, 1, 10)
        x = np.sin(np.linspace(0, 1, 15)).reshape(-1, 1)

        with pytest.raises(ValueError, match="incompatible"):
            solver.fit(t, x)

    def test_predict_with_invalid_input(self):
        """Test predict with invalid input type."""
        t = np.linspace(0, 1, 10)
        x = np.sin(t).reshape(-1, 1)

        solver = RK4Solver()
        solver.fit(t, x)

        with pytest.raises(TypeError):
            solver.predict([0, 1, 2])

    def test_predict_with_nan_in_t_eval(self):
        """Test predict rejects NaN in evaluation times."""
        t = np.linspace(0, 1, 10)
        x = np.sin(t).reshape(-1, 1)

        solver = RK4Solver()
        solver.fit(t, x)

        t_eval = np.linspace(0, 1, 10)
        t_eval[5] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            solver.predict(t_eval)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_state_variable(self):
        """Test solver with univariate ODE (1D state)."""
        t = np.linspace(0, 1, 20)
        x = np.exp(-t)

        solver = RK4Solver()
        solver.fit(t, x)
        x_pred = solver.predict(t)

        assert x_pred.ndim == 1, "1D input should produce 1D output"
        assert len(x_pred) == len(t)

    def test_two_state_variables(self):
        """Test solver with bivariate ODE (2D state)."""
        t = np.linspace(0, 2 * np.pi, 50)
        problem = harmonic_oscillator(t, x0=1.0, v0=0.0, omega=1.0)
        x = problem["x_exact"]

        solver = RK4Solver()
        solver.fit(t, x)
        x_pred = solver.predict(t)

        assert x_pred.ndim == 2, "2D input should produce 2D output"
        assert x_pred.shape[1] == 2

    def test_zero_initial_condition(self):
        """Test solver behavior with zero initial condition."""
        t = np.linspace(0, 1, 20)
        x = np.zeros((len(t), 1))

        solver = RK4Solver()
        solver.fit(t, x)
        x_pred = solver.predict(t)

        assert x_pred.shape == x.shape
        assert np.allclose(x_pred, 0, atol=1e-6)

    def test_constant_initial_condition(self):
        """Test solver with constant initial condition."""
        t = np.linspace(0, 1, 20)
        x = np.full((len(t), 1), 5.0)

        solver = RK4Solver()
        solver.fit(t, x)
        x_pred = solver.predict(t)

        assert x_pred.shape == x.shape
        assert np.allclose(x_pred, 5.0, atol=0.1)

    def test_dense_time_evaluation(self):
        """Test prediction on finer time grid than training."""
        t_train = np.linspace(0, 5, 20)
        problem = exponential_decay(t_train, x0=1.0, lambda_=0.5)
        x_train = problem["x_exact"]

        t_eval = np.linspace(0, 5, 100)

        solver = RK4Solver()
        solver.fit(t_train, x_train)
        x_pred = solver.predict(t_eval)

        assert x_pred.shape[0] == len(t_eval)

    def test_sparse_time_evaluation(self):
        """Test prediction on sparser time grid than training."""
        t_train = np.linspace(0, 5, 100)
        problem = exponential_decay(t_train, x0=1.0, lambda_=0.5)
        x_train = problem["x_exact"]

        t_eval = np.linspace(0, 5, 10)

        solver = RK4Solver()
        solver.fit(t_train, x_train)
        x_pred = solver.predict(t_eval)

        assert x_pred.shape[0] == len(t_eval)


class TestRK45SpecificFeatures:
    """Tests specific to RK45Solver adaptive stepping."""

    def test_rk45_tolerance_parameters(self, exponential_clean_data):
        """Test RK45Solver with custom tolerance parameters."""
        t, x_exact, params = exponential_clean_data

        solver = RK45Solver(rtol=1e-6, atol=1e-9)
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.005, (
            f"Tighter tolerances should improve accuracy, "
            f"RMSE={rmse_val}"
        )

    def test_rk45_loose_tolerance(self, exponential_clean_data):
        """Test RK45Solver with loose tolerance parameters."""
        t, x_exact, params = exponential_clean_data

        solver = RK45Solver(rtol=1e-2, atol=1e-3)
        solver.fit(t, x_exact)
        x_pred = solver.predict(t)

        rmse_val = rmse(x_exact, x_pred)
        assert rmse_val < 0.1, (
            f"Even loose tolerances should maintain reasonable accuracy, "
            f"RMSE={rmse_val}"
        )


class TestMultipleDimensions:
    """Tests for handling multi-dimensional systems."""

    def test_three_dimensional_system(self):
        """Test solver on a 3D system."""
        t = np.linspace(0, 1, 30)
        x = np.column_stack([
            np.exp(-0.5 * t),
            np.exp(-1.0 * t),
            np.exp(-2.0 * t)
        ])

        solver = RK4Solver()
        solver.fit(t, x)
        x_pred = solver.predict(t)

        assert x_pred.shape == (len(t), 3)
        rmse_val = rmse(x, x_pred)
        assert rmse_val < 0.05

    def test_high_dimensional_system(self):
        """Test solver on a 5D system."""
        t = np.linspace(0, 1, 50)
        x = np.column_stack([
            np.exp(-0.1 * i * t) for i in range(1, 6)
        ])

        solver = RK45Solver()
        solver.fit(t, x)
        x_pred = solver.predict(t)

        assert x_pred.shape == (len(t), 5)
        metrics = compute_all_metrics(x, x_pred)
        assert metrics["r_squared"] > 0.95
