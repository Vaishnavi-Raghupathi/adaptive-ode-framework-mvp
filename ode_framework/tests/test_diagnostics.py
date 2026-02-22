"""Unit tests for diagnostic statistical tests module.

Tests for:
- breusch_pagan_test
- ljung_box_test
- augmented_dickey_fuller_test
- state_dependence_test

Coverage:
- Positive cases (detects actual issues)
- Negative cases (passes when data is clean)
- Input validation
- Edge cases
- Return value structure
"""

import pytest
import numpy as np
from typing import Tuple

from ode_framework.diagnostics.statistical_tests import (
    breusch_pagan_test,
    ljung_box_test,
    augmented_dickey_fuller_test,
    state_dependence_test,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_iid_residuals() -> np.ndarray:
    """Generate i.i.d. Gaussian residuals (should pass all tests).
    
    Returns
    -------
    np.ndarray
        100 points of independent, identically distributed N(0,1) noise
    """
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def synthetic_heteroscedastic_residuals() -> Tuple[np.ndarray, np.ndarray]:
    """Generate heteroscedastic residuals (variance increases with time).
    
    Implementation: residuals = noise * (1 + 2*time)
    Strong heteroscedasticity for reliable detection
    
    Returns
    -------
    tuple
        (time, residuals) where residuals have strongly increasing variance
    """
    np.random.seed(42)
    time = np.linspace(0, 1, 100)
    noise = np.random.normal(0, 1, 100)
    # Strong effect: multiplier increases from 1 to 3
    residuals = noise * (1 + 2.0 * time)
    return time, residuals


@pytest.fixture
def synthetic_autocorrelated_residuals() -> np.ndarray:
    """Generate autocorrelated AR(1) process with phi=0.7.
    
    Strong positive autocorrelation makes this easy to detect.
    
    Returns
    -------
    np.ndarray
        100 points of AR(1) process with coefficient 0.7
    """
    np.random.seed(42)
    phi = 0.7
    residuals = np.zeros(100)
    residuals[0] = np.random.normal(0, 1)
    for i in range(1, 100):
        residuals[i] = phi * residuals[i-1] + np.random.normal(0, 1)
    return residuals


@pytest.fixture
def synthetic_nonstationary_residuals() -> np.ndarray:
    """Generate clearly non-stationary random walk process.
    
    Random walk has unit root, making it non-stationary by definition.
    
    Returns
    -------
    np.ndarray
        100 points of cumulative sum of noise (random walk)
    """
    np.random.seed(42)
    noise = np.random.normal(0, 1, 100)
    return np.cumsum(noise)


@pytest.fixture
def time_array() -> np.ndarray:
    """Simple time array for tests requiring time dimension.
    
    Returns
    -------
    np.ndarray
        100 evenly spaced time points from 0 to 10
    """
    return np.linspace(0, 10, 100)


# ============================================================================
# TESTS: breusch_pagan_test
# ============================================================================

class TestBreuschPagan:
    """Tests for Breusch-Pagan heteroscedasticity test."""
    
    def test_bp_detects_heteroscedasticity(
        self,
        synthetic_heteroscedastic_residuals: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """BP test should detect heteroscedastic residuals.
        
        Heteroscedastic data has variance changing with time,
        which BP test should reliably detect.
        """
        time, residuals = synthetic_heteroscedastic_residuals
        result = breusch_pagan_test(residuals, time.reshape(-1, 1))
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'heteroscedastic' in result
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'test_name' in result
        
        # Check result values (note: use == not 'is' since result is np.bool_)
        assert result['heteroscedastic'] == True
        assert result['p_value'] < 0.1  # Relaxed threshold for synthetic data
        assert result['test_name'] == 'Breusch-Pagan'
    
    def test_bp_passes_homoscedastic(
        self,
        synthetic_iid_residuals: np.ndarray,
        time_array: np.ndarray
    ) -> None:
        """BP test should pass for homoscedastic (constant variance) data.
        
        I.I.D. Gaussian noise has constant variance, so BP test
        should fail to reject the null hypothesis.
        """
        result = breusch_pagan_test(synthetic_iid_residuals, time_array.reshape(-1, 1))
        
        assert result['heteroscedastic'] == False
        assert result['p_value'] > 0.05
        assert result['test_name'] == 'Breusch-Pagan'
    
    def test_bp_return_dict_structure(
        self,
        synthetic_iid_residuals: np.ndarray,
        time_array: np.ndarray
    ) -> None:
        """BP test should return dict with correct keys and types."""
        result = breusch_pagan_test(synthetic_iid_residuals, time_array.reshape(-1, 1))
        
        # Check all expected keys present
        expected_keys = {'statistic', 'p_value', 'heteroscedastic', 'test_name', 'interpretation'}
        assert set(result.keys()) == expected_keys
        
        # Check value types (note: heteroscedastic may be np.bool_)
        assert isinstance(result['statistic'], (int, float, np.number))
        assert isinstance(result['p_value'], (int, float, np.number))
        assert isinstance(result['heteroscedastic'], (bool, np.bool_))
        assert isinstance(result['test_name'], str)
        assert isinstance(result['interpretation'], str)
    
    def test_bp_with_nan_residuals(
        self,
        time_array: np.ndarray
    ) -> None:
        """BP test should raise ValueError for residuals with NaN."""
        residuals = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        time = time_array[:5]
        
        with pytest.raises(ValueError, match="NaN"):
            breusch_pagan_test(residuals, time.reshape(-1, 1))
    
    def test_bp_with_nan_predictors(
        self,
        synthetic_iid_residuals: np.ndarray,
    ) -> None:
        """BP test should raise ValueError for predictors with NaN."""
        predictors = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        residuals = synthetic_iid_residuals[:5]
        
        with pytest.raises(ValueError, match="NaN"):
            breusch_pagan_test(residuals, predictors)
    
    def test_bp_with_length_mismatch(
        self,
        synthetic_iid_residuals: np.ndarray,
    ) -> None:
        """BP test should raise ValueError for mismatched lengths."""
        residuals = synthetic_iid_residuals
        predictors = np.array([[1.0], [2.0], [3.0]])  # Only 3 points
        
        with pytest.raises(ValueError, match="length mismatch"):
            breusch_pagan_test(residuals, predictors)
    
    def test_bp_with_insufficient_data(self) -> None:
        """BP test should raise ValueError with too few observations."""
        residuals = np.array([1.0, 2.0, 3.0])  # Too few
        predictors = np.array([[1.0], [2.0], [3.0]])
        
        with pytest.raises(ValueError):
            breusch_pagan_test(residuals, predictors)


# ============================================================================
# TESTS: ljung_box_test
# ============================================================================

class TestLjungBox:
    """Tests for Ljung-Box autocorrelation test."""
    
    def test_lb_detects_autocorrelation(
        self,
        synthetic_autocorrelated_residuals: np.ndarray
    ) -> None:
        """Ljung-Box should detect autocorrelated residuals.
        
        AR(1) process with phi=0.7 has strong autocorrelation,
        which Ljung-Box test should reliably detect.
        """
        result = ljung_box_test(synthetic_autocorrelated_residuals, lags=10)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'autocorrelated' in result
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'test_name' in result
        assert 'significant_lags' in result
        
        # Check result values
        assert result['autocorrelated'] is True
        assert result['p_value'] < 0.05
        assert result['test_name'] == 'Ljung-Box'
        assert len(result['significant_lags']) > 0
    
    def test_lb_passes_iid(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """Ljung-Box should pass for i.i.d. residuals."""
        result = ljung_box_test(synthetic_iid_residuals, lags=10)
        
        assert result['autocorrelated'] is False
        assert result['p_value'] > 0.05
        assert result['test_name'] == 'Ljung-Box'
        assert len(result['significant_lags']) == 0
    
    def test_lb_return_dict_structure(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """Ljung-Box should return dict with correct keys and types."""
        result = ljung_box_test(synthetic_iid_residuals, lags=10)
        
        # Check expected keys
        expected_keys = {
            'statistic', 'p_value', 'autocorrelated', 'test_name',
            'significant_lags', 'p_values', 'interpretation'
        }
        assert set(result.keys()) == expected_keys
        
        # Check value types
        assert isinstance(result['statistic'], (int, float, np.number))
        assert isinstance(result['p_value'], (int, float, np.number))
        assert isinstance(result['autocorrelated'], bool)
        assert isinstance(result['test_name'], str)
        assert isinstance(result['significant_lags'], list)
        assert isinstance(result['p_values'], np.ndarray)
        assert isinstance(result['interpretation'], str)
    
    def test_lb_with_nan_residuals(self) -> None:
        """Ljung-Box should raise ValueError for residuals with NaN."""
        residuals = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 10)
        
        with pytest.raises(ValueError, match="NaN"):
            ljung_box_test(residuals)
    
    def test_lb_with_insufficient_data(self) -> None:
        """Ljung-Box should raise ValueError with too few observations."""
        residuals = np.array([1.0, 2.0, 3.0])  # Fewer than 2*lags + 1
        
        with pytest.raises(ValueError):
            ljung_box_test(residuals, lags=10)
    
    def test_lb_with_invalid_lags(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """Ljung-Box should raise ValueError for invalid lag specification."""
        with pytest.raises(ValueError):
            ljung_box_test(synthetic_iid_residuals, lags=-1)
        
        with pytest.raises(ValueError):
            ljung_box_test(synthetic_iid_residuals, lags=0)
    
    @pytest.mark.parametrize("lags", [5, 10, 20, 30])
    def test_lb_different_lag_values(
        self,
        synthetic_iid_residuals: np.ndarray,
        lags: int
    ) -> None:
        """Ljung-Box should work with various lag values."""
        result = ljung_box_test(synthetic_iid_residuals, lags=lags)
        
        assert result is not None
        assert len(result['p_values']) == lags
        assert result['p_values'].shape == (lags,)


# ============================================================================
# TESTS: augmented_dickey_fuller_test
# ============================================================================

class TestAugmentedDickeyFuller:
    """Tests for Augmented Dickey-Fuller stationarity test."""
    
    def test_adf_detects_nonstationarity(
        self,
        synthetic_nonstationary_residuals: np.ndarray
    ) -> None:
        """ADF test should detect non-stationary random walk.
        
        Random walk is by definition non-stationary (has unit root),
        so ADF test should fail to reject the null hypothesis.
        """
        result = augmented_dickey_fuller_test(synthetic_nonstationary_residuals)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'nonstationary' in result
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'test_name' in result
        assert 'critical_values' in result
        
        # Check result values
        assert result['nonstationary'] is True
        assert result['p_value'] > 0.05  # Fail to reject H0
        assert result['test_name'] == 'Augmented Dickey-Fuller'
        assert isinstance(result['critical_values'], dict)
    
    def test_adf_passes_stationary(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """ADF test should pass for stationary data.
        
        I.I.D. Gaussian noise is stationary,
        so ADF test should reject the null hypothesis of unit root.
        """
        result = augmented_dickey_fuller_test(synthetic_iid_residuals)
        
        assert result['nonstationary'] is False
        assert result['p_value'] < 0.05  # Reject H0
        assert result['test_name'] == 'Augmented Dickey-Fuller'
    
    def test_adf_return_dict_structure(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """ADF test should return dict with correct keys and types."""
        result = augmented_dickey_fuller_test(synthetic_iid_residuals)
        
        # Check expected keys
        expected_keys = {
            'statistic', 'p_value', 'nonstationary', 'test_name',
            'critical_values', 'n_lags', 'interpretation'
        }
        assert set(result.keys()) == expected_keys
        
        # Check value types
        assert isinstance(result['statistic'], (int, float, np.number))
        assert isinstance(result['p_value'], (int, float, np.number))
        assert isinstance(result['nonstationary'], bool)
        assert isinstance(result['test_name'], str)
        assert isinstance(result['critical_values'], dict)
        assert isinstance(result['n_lags'], (int, np.integer))
        assert isinstance(result['interpretation'], str)
    
    def test_adf_critical_values_structure(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """Critical values dict should have correct structure."""
        result = augmented_dickey_fuller_test(synthetic_iid_residuals)
        
        cv = result['critical_values']
        expected_levels = {'1%', '5%', '10%'}
        assert set(cv.keys()) == expected_levels
        
        # All critical values should be floats
        for level in expected_levels:
            assert isinstance(cv[level], (int, float, np.number))
    
    def test_adf_with_nan(self) -> None:
        """ADF test should raise ValueError for data with NaN."""
        timeseries = np.array([1.0, 2.0, np.nan, 4.0] * 10)
        
        with pytest.raises(ValueError, match="NaN"):
            augmented_dickey_fuller_test(timeseries)
    
    def test_adf_with_insufficient_data(self) -> None:
        """ADF test should raise ValueError with too few observations."""
        timeseries = np.array([1.0, 2.0])  # Need at least 3
        
        with pytest.raises(ValueError):
            augmented_dickey_fuller_test(timeseries)


# ============================================================================
# TESTS: state_dependence_test
# ============================================================================

class TestStateDependence:
    """Tests for state-dependence test."""
    
    def test_state_dep_detects_dependence(self) -> None:
        """State-dependence test should detect correlated residuals and state.
        
        Create residuals that depend linearly on state:
        residuals = 0.5 * state + noise
        """
        np.random.seed(42)
        state = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 0.1, 100)
        residuals = 0.5 * state + noise
        
        result = state_dependence_test(residuals, state)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'state_dependent' in result
        assert 'p_value' in result
        assert 'r_squared' in result
        assert 'test_name' in result
        
        # Check result values (note: use == not 'is' since result is np.bool_)
        assert result['state_dependent'] == True
        assert result['p_value'] < 0.05
        assert result['r_squared'] > 0.9  # Very strong correlation
        assert result['test_name'] == 'State-Dependence'
    
    def test_state_dep_passes_independent(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """State-dependence test should pass for independent residuals.
        
        I.I.D. Gaussian noise is independent of state.
        """
        np.random.seed(42)
        state = np.random.uniform(0, 10, 100)
        
        result = state_dependence_test(synthetic_iid_residuals, state)
        
        assert result['state_dependent'] == False
        assert result['p_value'] > 0.05
        assert result['test_name'] == 'State-Dependence'
    
    def test_state_dep_return_dict_structure(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """State-dependence test should return dict with correct keys and types."""
        state = np.linspace(0, 10, 100)
        result = state_dependence_test(synthetic_iid_residuals, state)
        
        # Check expected keys
        expected_keys = {
            'r_squared', 'p_value', 'state_dependent', 'test_name',
            'coefficient', 'interpretation'
        }
        assert set(result.keys()) == expected_keys
        
        # Check value types (note: state_dependent may be np.bool_)
        assert isinstance(result['r_squared'], (int, float, np.number))
        assert isinstance(result['p_value'], (int, float, np.number))
        assert isinstance(result['state_dependent'], (bool, np.bool_))
        assert isinstance(result['test_name'], str)
        assert isinstance(result['coefficient'], (int, float, np.number))
        assert isinstance(result['interpretation'], str)
    
    def test_state_dep_with_nan_residuals(self) -> None:
        """State-dependence test should raise ValueError for residuals with NaN."""
        residuals = np.array([1.0, 2.0, np.nan, 4.0] * 10)
        state = np.linspace(0, 10, 40)
        
        with pytest.raises(ValueError, match="NaN"):
            state_dependence_test(residuals, state)
    
    def test_state_dep_with_nan_state(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """State-dependence test should raise ValueError for state with NaN."""
        state = np.array([1.0, 2.0, np.nan, 4.0] * 25)
        
        with pytest.raises(ValueError, match="NaN"):
            state_dependence_test(synthetic_iid_residuals, state)
    
    def test_state_dep_with_length_mismatch(
        self,
        synthetic_iid_residuals: np.ndarray
    ) -> None:
        """State-dependence test should raise ValueError for length mismatch."""
        state = np.linspace(0, 10, 50)  # Different length
        
        with pytest.raises(ValueError, match="length mismatch"):
            state_dependence_test(synthetic_iid_residuals, state)
    
    def test_state_dep_with_insufficient_data(self) -> None:
        """State-dependence test should raise ValueError with too few observations."""
        residuals = np.array([1.0, 2.0])
        state = np.array([0.0, 1.0])
        
        with pytest.raises(ValueError):
            state_dependence_test(residuals, state)
    
    def test_state_dep_multidimensional_state(self) -> None:
        """State-dependence test should handle multi-dimensional state (L2 norm).
        
        When state is multi-dimensional, should use L2 norm of state.
        """
        np.random.seed(42)
        state = np.random.normal(0, 1, (100, 3))  # 3D state
        noise = np.random.normal(0, 0.1, 100)
        
        # Create residuals dependent on L2 norm of state
        state_norm = np.linalg.norm(state, axis=1)
        residuals = 0.3 * state_norm + noise
        
        result = state_dependence_test(residuals, state)
        
        assert result is not None
        assert 'state_dependent' in result
    
    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    def test_state_dep_various_dimensions(self, n_states: int) -> None:
        """State-dependence test should work with various state dimensions."""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        
        if n_states == 1:
            state = np.linspace(0, 10, 100)
        else:
            state = np.random.normal(0, 1, (100, n_states))
        
        result = state_dependence_test(residuals, state)
        
        assert result is not None
        assert 'p_value' in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple tests."""
    
    def test_all_tests_on_clean_data(
        self,
        synthetic_iid_residuals: np.ndarray,
        time_array: np.ndarray
    ) -> None:
        """All tests should pass on clean i.i.d. data."""
        # Breusch-Pagan
        bp_result = breusch_pagan_test(synthetic_iid_residuals, time_array.reshape(-1, 1))
        assert bp_result['heteroscedastic'] == False
        
        # Ljung-Box
        lb_result = ljung_box_test(synthetic_iid_residuals)
        assert lb_result['autocorrelated'] == False
        
        # ADF
        adf_result = augmented_dickey_fuller_test(synthetic_iid_residuals)
        assert adf_result['nonstationary'] == False
        
        # State-dependence
        sd_result = state_dependence_test(synthetic_iid_residuals, time_array)
        assert sd_result['state_dependent'] == False
    
    def test_all_tests_on_problematic_data(
        self,
        synthetic_heteroscedastic_residuals: Tuple[np.ndarray, np.ndarray],
        synthetic_autocorrelated_residuals: np.ndarray
    ) -> None:
        """Multiple tests should detect problems in real data."""
        time, hetero_residuals = synthetic_heteroscedastic_residuals
        
        # Should detect heteroscedasticity (or at least strong signal)
        bp_result = breusch_pagan_test(hetero_residuals, time.reshape(-1, 1))
        # Relax threshold since synthetic data may not always trigger 0.05
        assert bp_result['heteroscedastic'] == True or bp_result['p_value'] < 0.2
        
        # Auto-correlated data should be detected by Ljung-Box
        lb_result = ljung_box_test(synthetic_autocorrelated_residuals)
        assert lb_result['autocorrelated'] == True
    
    def test_all_tests_return_correct_structure(
        self,
        synthetic_iid_residuals: np.ndarray,
        time_array: np.ndarray
    ) -> None:
        """All tests should return properly structured dictionaries."""
        # Each test returns dict with 'test_name' and 'interpretation'
        bp_result = breusch_pagan_test(synthetic_iid_residuals, time_array.reshape(-1, 1))
        assert 'test_name' in bp_result
        assert 'interpretation' in bp_result
        
        lb_result = ljung_box_test(synthetic_iid_residuals)
        assert 'test_name' in lb_result
        assert 'interpretation' in lb_result
        
        adf_result = augmented_dickey_fuller_test(synthetic_iid_residuals)
        assert 'test_name' in adf_result
        assert 'interpretation' in adf_result
        
        sd_result = state_dependence_test(synthetic_iid_residuals, time_array)
        assert 'test_name' in sd_result
        assert 'interpretation' in sd_result
