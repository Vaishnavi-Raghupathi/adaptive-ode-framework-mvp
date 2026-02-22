"""Statistical tests for ODE solver diagnostics.

This module provides statistical tests to validate the quality of ODE solver
fits and predictions. These tests help identify potential issues with the
fitted model, such as autocorrelation in residuals, heteroscedasticity, and
non-stationarity in the underlying system.

Tests Implemented:
    - Breusch-Pagan Test: Tests for heteroscedasticity (non-constant variance)
    - Ljung-Box Test: Tests for autocorrelation in residuals
    - Augmented Dickey-Fuller Test: Tests for stationarity
    - State Dependence Test: Tests for dependence on current state
    
Example:
    >>> import numpy as np
    >>> from ode_framework.diagnostics.statistical_tests import breusch_pagan_test
    >>> residuals = np.random.normal(0, 1, 100)
    >>> state = np.random.normal(0, 1, 100)
    >>> result = breusch_pagan_test(residuals, state)
    >>> print(f"Heteroscedastic: {result['heteroscedastic']}")
"""

from typing import Dict, Any
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def breusch_pagan_test(residuals: np.ndarray, predictors: np.ndarray) -> Dict[str, Any]:
    """Test for heteroscedasticity in residuals.
    
    The Breusch-Pagan test checks if the variance of residuals depends on
    the predictor variables (time, state, etc.). Heteroscedasticity (non-constant
    variance) can indicate that the model's prediction error changes systematically
    with the system state, suggesting model misspecification or unmodeled dynamics.
    
    Mathematical Background:
        1. Compute squared residuals: u_i^2
        2. Regress u_i^2 on predictors: u_i^2 = α + β*X_i + ε_i
        3. Test H0: β = 0 (homoscedasticity) vs H1: β ≠ 0
        4. Test statistic ~ χ²(k) where k = number of predictors
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from ODE solver fit. Shape: (n_samples,) or (n_samples, 1)
    predictors : np.ndarray
        Predictor variables (time or state). Shape: (n_samples,) or (n_samples, p)
        where p is number of predictor variables.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': float, test statistic (chi-squared distributed)
        - 'p_value': float, p-value for the test
        - 'heteroscedastic': bool, True if p_value < 0.05
        - 'test_name': str, 'Breusch-Pagan'
        - 'interpretation': str, explanation of result
    
    Raises
    ------
    ValueError
        If residuals or predictors contain NaN/Inf or have shape mismatch
    RuntimeError
        If test computation fails
    
    Notes
    -----
    - H0 (Null Hypothesis): Residuals have constant variance
    - H1 (Alternative): Residuals variance depends on predictors
    - Interpretation:
      - p_value < 0.05: Reject H0, evidence of heteroscedasticity
      - p_value >= 0.05: Fail to reject H0, constant variance likely
    - Result suggests model improvements needed if heteroscedasticity detected
    
    References
    ----------
    Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity
    and random coefficient variation. Econometrica, 47(5), 1287-1294.
    
    Examples
    --------
    >>> import numpy as np
    >>> residuals = np.random.normal(0, 1, 100)
    >>> predictors = np.linspace(0, 10, 100)
    >>> result = breusch_pagan_test(residuals, predictors)
    >>> print(f"Heteroscedastic: {result['heteroscedastic']}")
    """
    # Input validation
    residuals = np.asarray(residuals).flatten()
    predictors = np.asarray(predictors)
    
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D array after flattening, got shape {residuals.shape}")
    
    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        raise ValueError("residuals contain NaN or Inf values")
    
    if predictors.ndim == 1:
        predictors = predictors.reshape(-1, 1)
    elif predictors.ndim != 2:
        raise ValueError(f"predictors must be 1D or 2D, got shape {predictors.shape}")
    
    if np.any(np.isnan(predictors)) or np.any(np.isinf(predictors)):
        raise ValueError("predictors contain NaN or Inf values")
    
    if len(residuals) != len(predictors):
        raise ValueError(
            f"residuals and predictors length mismatch: {len(residuals)} vs {len(predictors)}"
        )
    
    # Need at least 2 observations per predictor
    min_samples = 2 * (predictors.shape[1] + 1)
    if len(residuals) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} observations for {predictors.shape[1]} predictors, "
            f"got {len(residuals)}"
        )
    
    try:
        # Add constant term to predictors (required by statsmodels)
        predictors_with_const = np.column_stack([np.ones(len(predictors)), predictors])
        
        # Run Breusch-Pagan test
        test_stat, p_value, _, _ = het_breuschpagan(residuals, predictors_with_const)
        
        # Determine if heteroscedastic
        heteroscedastic = p_value < 0.05
        
        # Generate interpretation
        if heteroscedastic:
            interpretation = (
                f"Evidence of heteroscedasticity detected (p={p_value:.4f}). "
                "Residual variance depends on predictors. Consider model refinement."
            )
        else:
            interpretation = (
                f"No significant heteroscedasticity (p={p_value:.4f}). "
                "Residual variance appears constant across predictors."
            )
        
        return {
            'statistic': float(test_stat),
            'p_value': float(p_value),
            'heteroscedastic': heteroscedastic,
            'test_name': 'Breusch-Pagan',
            'interpretation': interpretation,
        }
    except Exception as e:
        raise RuntimeError(f"Breusch-Pagan test failed: {str(e)}") from e


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
    """Test for autocorrelation in residuals using Ljung-Box test.
    
    The Ljung-Box test checks if residuals exhibit autocorrelation (serial
    correlation). Significant autocorrelation suggests the ODE model has not
    captured all temporal dependencies, indicating potential model misspecification.
    
    Mathematical Background:
        Q = n(n+2) * Σ(ρ̂_k^2 / (n-k)) for k=1..h lags
        where ρ̂_k is the sample autocorrelation at lag k
        Q ~ χ²(h) under H0 of independence
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from ODE solver fit. Shape: (n_samples,) or (n_samples, 1)
    lags : int, optional
        Number of lags to test for autocorrelation. Default is 10.
        Should be less than n_samples / 2.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': float, Ljung-Box test statistic
        - 'p_value': float, p-value for the test
        - 'autocorrelated': bool, True if any lag shows significant autocorrelation
        - 'test_name': str, 'Ljung-Box'
        - 'significant_lags': list of int, lags with p_value < 0.05
        - 'p_values': np.ndarray, p-values for each lag
        - 'interpretation': str, explanation of result
    
    Raises
    ------
    ValueError
        If residuals contain NaN/Inf or have insufficient data
    RuntimeError
        If test computation fails
    
    Notes
    -----
    - H0 (Null Hypothesis): Residuals are independently distributed
    - H1 (Alternative): Residuals exhibit autocorrelation
    - Interpretation:
      - p_value < 0.05: Reject H0, evidence of autocorrelation
      - p_value >= 0.05: Fail to reject H0, independence likely
    - Autocorrelation indicates the model missed temporal structure
    - Need at least 2*lags + 1 observations for reliable test
    
    References
    ----------
    Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in time
    series models. Biometrika, 65(2), 297-303.
    
    Examples
    --------
    >>> import numpy as np
    >>> residuals = np.random.normal(0, 1, 100)
    >>> result = ljung_box_test(residuals, lags=10)
    >>> print(f"Autocorrelated: {result['autocorrelated']}")
    >>> print(f"Significant lags: {result['significant_lags']}")
    """
    # Input validation
    residuals = np.asarray(residuals).flatten()
    
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D array after flattening, got shape {residuals.shape}")
    
    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        raise ValueError("residuals contain NaN or Inf values")
    
    if not isinstance(lags, int) or lags < 1:
        raise ValueError(f"lags must be positive integer, got {lags}")
    
    # Check sufficient data
    min_samples = 2 * lags + 1
    if len(residuals) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} observations for {lags} lags, got {len(residuals)}"
        )
    
    try:
        # Run Ljung-Box test
        # Note: acorr_ljungbox returns tuple (test_stats, p_values) when return_df=False
        # But with newer statsmodels, it returns a DataFrame. We need to handle both.
        result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        
        # Extract p_values from DataFrame
        p_values = result['lb_pvalue'].values
        test_stats = result['lb_stat'].values
        
        # Find significant lags
        significant_lags = np.where(p_values < 0.05)[0].tolist()
        autocorrelated = len(significant_lags) > 0
        
        # Use overall test statistic (last lag)
        overall_stat = float(test_stats[-1])
        overall_p = float(p_values[-1])
        
        # Generate interpretation
        if autocorrelated:
            interpretation = (
                f"Autocorrelation detected at lags {significant_lags} (p={overall_p:.4f}). "
                "ODE model may not capture all temporal dynamics."
            )
        else:
            interpretation = (
                f"No significant autocorrelation detected (p={overall_p:.4f}). "
                "Residuals appear to be independently distributed."
            )
        
        return {
            'statistic': overall_stat,
            'p_value': overall_p,
            'autocorrelated': autocorrelated,
            'test_name': 'Ljung-Box',
            'significant_lags': significant_lags,
            'p_values': p_values,
            'interpretation': interpretation,
        }
    except Exception as e:
        raise RuntimeError(f"Ljung-Box test failed: {str(e)}") from e


def augmented_dickey_fuller_test(timeseries: np.ndarray) -> Dict[str, Any]:
    """Test for stationarity of time series using Augmented Dickey-Fuller test.
    
    The Augmented Dickey-Fuller (ADF) test checks if a time series is stationary
    (constant mean and variance). Non-stationary systems may indicate the ODE
    model needs reparameterization or that external forces are changing system
    properties over time.
    
    Mathematical Background:
        Tests for unit root in: Δy_t = α + β*t + γ*y_{t-1} + Σδ_i*Δy_{t-i} + ε_t
        where Δ is the difference operator
        H0: γ = 0 (unit root, non-stationary)
        H1: γ < 0 (stationary)
        Test statistic follows Dickey-Fuller distribution
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series to test for stationarity. Shape: (n_samples,) or (n_samples, 1)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': float, ADF test statistic
        - 'p_value': float, p-value for the test
        - 'nonstationary': bool, True if p_value > 0.05 (fail to reject H0)
        - 'test_name': str, 'Augmented Dickey-Fuller'
        - 'critical_values': dict, critical values at 1%, 5%, 10% levels
        - 'n_lags': int, number of lags used in regression
        - 'interpretation': str, explanation of result
    
    Raises
    ------
    ValueError
        If timeseries contains NaN/Inf or has insufficient data
    RuntimeError
        If test computation fails
    
    Notes
    -----
    - H0 (Null Hypothesis): Time series has a unit root (non-stationary)
    - H1 (Alternative): Time series is stationary
    - Interpretation:
      - p_value < 0.05: Reject H0, time series likely stationary
      - p_value >= 0.05: Fail to reject H0, time series may be non-stationary
    - Non-stationarity suggests external forcing or time-varying parameters
    - Need at least 3 observations for reliable test
    
    References
    ----------
    Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
    for autoregressive time series with a unit root. Journal of the American
    Statistical Association, 74(366), 427-431.
    
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> timeseries = np.exp(-0.5 * t) * np.sin(t)  # Decaying oscillation (stationary)
    >>> result = augmented_dickey_fuller_test(timeseries)
    >>> print(f"Nonstationary: {result['nonstationary']}")
    """
    # Input validation
    timeseries = np.asarray(timeseries).flatten()
    
    if timeseries.ndim != 1:
        raise ValueError(f"timeseries must be 1D array after flattening, got shape {timeseries.shape}")
    
    if np.any(np.isnan(timeseries)) or np.any(np.isinf(timeseries)):
        raise ValueError("timeseries contains NaN or Inf values")
    
    if len(timeseries) < 3:
        raise ValueError(f"Need at least 3 observations, got {len(timeseries)}")
    
    try:
        # Run ADF test
        result = adfuller(timeseries, autolag='AIC')
        
        adf_stat = float(result[0])
        p_value = float(result[1])
        n_lags = int(result[2])
        critical_values = {
            '1%': float(result[4]['1%']),
            '5%': float(result[4]['5%']),
            '10%': float(result[4]['10%']),
        }
        
        # Determine if nonstationary
        # Fail to reject H0 (non-stationary) if p_value > 0.05
        nonstationary = p_value > 0.05
        
        # Generate interpretation
        if nonstationary:
            interpretation = (
                f"Series appears non-stationary (p={p_value:.4f}). "
                "May indicate time-varying system properties or external forcing."
            )
        else:
            interpretation = (
                f"Series appears stationary (p={p_value:.4f}). "
                "Mean and variance likely constant over time."
            )
        
        return {
            'statistic': adf_stat,
            'p_value': p_value,
            'nonstationary': nonstationary,
            'test_name': 'Augmented Dickey-Fuller',
            'critical_values': critical_values,
            'n_lags': n_lags,
            'interpretation': interpretation,
        }
    except Exception as e:
        raise RuntimeError(f"Augmented Dickey-Fuller test failed: {str(e)}") from e


def state_dependence_test(residuals: np.ndarray, state_vars: np.ndarray) -> Dict[str, Any]:
    """Test for dependence of residuals on state variables.
    
    Tests whether the magnitude of residuals systematically depends on the
    current state, which would indicate model misspecification or nonlinear
    effects not captured by the identified ODE.
    
    Mathematical Background:
        Regress |residuals| on state_vars using linear regression
        |u_i| = α + β*X_i + ε_i
        H0: β = 0 (residuals independent of state)
        H1: β ≠ 0 (residuals depend on state)
        Uses t-test for coefficient significance
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from ODE solver fit. Shape: (n_samples,) or (n_samples, 1)
    state_vars : np.ndarray
        State variable(s) at each time point. Shape: (n_samples,) or (n_samples, n_states)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'r_squared': float, coefficient of determination
        - 'p_value': float, p-value for the test
        - 'state_dependent': bool, True if p_value < 0.05
        - 'test_name': str, 'State-Dependence'
        - 'coefficient': float, regression coefficient
        - 'interpretation': str, explanation of result
    
    Raises
    ------
    ValueError
        If residuals or state_vars contain NaN/Inf or have shape mismatch
    RuntimeError
        If test computation fails
    
    Notes
    -----
    - H0 (Null Hypothesis): Residuals are independent of state
    - H1 (Alternative): Residuals depend on state
    - Interpretation:
      - p_value < 0.05: Reject H0, evidence of state dependence
      - p_value >= 0.05: Fail to reject H0, independence likely
    - High dependence indicates nonlinear dynamics not captured
    - Uses absolute value of residuals (magnitude of error)
    - For multi-dimensional state, uses L2 norm
    
    Methods
    -------
    Uses Pearson correlation test with scipy.stats.linregress.
    
    Examples
    --------
    >>> import numpy as np
    >>> residuals = np.random.normal(0, 1, 100)
    >>> state = np.linspace(0, 10, 100)
    >>> result = state_dependence_test(residuals, state)
    >>> print(f"State dependent: {result['state_dependent']}")
    """
    # Input validation
    residuals = np.asarray(residuals).flatten()
    state_vars = np.asarray(state_vars)
    
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D array after flattening, got shape {residuals.shape}")
    
    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        raise ValueError("residuals contain NaN or Inf values")
    
    if state_vars.ndim == 1:
        state_vars = state_vars.reshape(-1, 1)
    elif state_vars.ndim != 2:
        raise ValueError(f"state_vars must be 1D or 2D, got shape {state_vars.shape}")
    
    if np.any(np.isnan(state_vars)) or np.any(np.isinf(state_vars)):
        raise ValueError("state_vars contain NaN or Inf values")
    
    if len(residuals) != len(state_vars):
        raise ValueError(
            f"residuals and state_vars length mismatch: {len(residuals)} vs {len(state_vars)}"
        )
    
    if len(residuals) < 3:
        raise ValueError(f"Need at least 3 observations, got {len(residuals)}")
    
    try:
        # Use absolute residuals as dependent variable
        abs_residuals = np.abs(residuals)
        
        # For multi-dimensional state, use L2 norm
        if state_vars.shape[1] > 1:
            state_magnitude = np.linalg.norm(state_vars, axis=1)
        else:
            state_magnitude = state_vars.flatten()
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(state_magnitude, abs_residuals)
        
        # Compute R-squared
        r_squared = r_value ** 2
        
        # Determine if state dependent
        state_dependent = p_value < 0.05
        
        # Generate interpretation
        if state_dependent:
            interpretation = (
                f"Residual magnitude depends on state (p={p_value:.4f}, R²={r_squared:.4f}). "
                "ODE may not capture state-dependent dynamics."
            )
        else:
            interpretation = (
                f"No significant state dependence (p={p_value:.4f}, R²={r_squared:.4f}). "
                "Residuals appear independent of state values."
            )
        
        return {
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'state_dependent': state_dependent,
            'test_name': 'State-Dependence',
            'coefficient': float(slope),
            'interpretation': interpretation,
        }
    except Exception as e:
        raise RuntimeError(f"State-dependence test failed: {str(e)}") from e
