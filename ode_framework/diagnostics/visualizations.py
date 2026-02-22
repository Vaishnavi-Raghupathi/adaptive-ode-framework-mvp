"""Visualization utilities for ODE solver diagnostics.

This module provides functions to create diagnostic plots and visualizations
that help understand solver behavior and identify potential issues. Plots
include residual diagnostics, convergence analysis, and solution comparisons.

Visualization Types:
    - Residual plots: Time series with zero line and distribution
    - ACF plots: Autocorrelation function with confidence bands
    - Q-Q plots: Normality assessment of residuals
    - Variance trends: Rolling variance to detect heteroscedasticity
    - Combined diagnostics: Multi-panel overview of all key metrics
    
Example:
    >>> from ode_framework.diagnostics.visualizations import plot_diagnostics
    >>> from ode_framework.solvers.classical import RK45Solver
    >>> solver = RK45Solver()
    >>> solver.fit(t_data, x_data)
    >>> plot_diagnostics(t_data, residuals, test_results)
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf


def plot_residuals_timeseries(
    time: np.ndarray,
    residuals: np.ndarray,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot residuals over time with zero reference line.
    
    Creates a scatter plot of residuals connected by lines with a horizontal
    zero reference line (dashed). Useful for identifying systematic patterns
    in residuals over time.
    
    Parameters
    ----------
    time : np.ndarray
        Time points. Shape: (n_samples,)
    residuals : np.ndarray
        Residuals from ODE solver. Shape: (n_samples,) or (n_samples, 1)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes object for further customization
    
    Notes
    -----
    Residuals should ideally:
    - Scatter randomly around zero line
    - Show no systematic trends
    - Have consistent spread (homoscedasticity)
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ode_framework.diagnostics.visualizations import plot_residuals_timeseries
    >>> time = np.linspace(0, 10, 100)
    >>> residuals = np.random.normal(0, 1, 100)
    >>> ax = plot_residuals_timeseries(time, residuals)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Convert to 1D array
    residuals = np.asarray(residuals).flatten()
    time = np.asarray(time).flatten()
    
    # Plot residuals
    ax.plot(time, residuals, 'o-', alpha=0.6, linewidth=1.5, markersize=4, color='#1f77b4')
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
    
    # Formatting
    ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax.set_title('Residual Time Series', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    return ax


def plot_acf(
    residuals: np.ndarray,
    lags: int = 20,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot autocorrelation function of residuals.
    
    Uses statsmodels to compute and plot autocorrelation function with
    confidence bands. Helps identify if residuals exhibit temporal dependence.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from ODE solver. Shape: (n_samples,) or (n_samples, 1)
    lags : int, optional
        Number of lags to display. Default is 20.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes object for further customization
    
    Notes
    -----
    Interpretation:
    - Bars within confidence interval (shaded region) indicate no significant autocorrelation
    - Bars extending beyond interval suggest autocorrelation at that lag
    - Significant autocorrelation indicates model misspecification
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ode_framework.diagnostics.visualizations import plot_acf
    >>> residuals = np.random.normal(0, 1, 100)
    >>> ax = plot_acf(residuals, lags=20)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Convert to 1D array
    residuals = np.asarray(residuals).flatten()
    
    # Ensure we don't ask for more lags than data points
    lags = min(lags, len(residuals) // 2 - 1)
    
    # Plot ACF using statsmodels
    sm_plot_acf(residuals, lags=lags, ax=ax)
    
    # Formatting
    ax.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=11, fontweight='bold')
    ax.set_ylabel('ACF', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_qq(
    residuals: np.ndarray,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Q-Q plot to assess normality of residuals.
    
    Plots theoretical normal quantiles against sample quantiles. Residuals
    that follow a normal distribution should fall near the 45-degree line.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from ODE solver. Shape: (n_samples,) or (n_samples, 1)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes object for further customization
    
    Notes
    -----
    Interpretation:
    - Points along diagonal line: Residuals approximately normal
    - Points deviate at tails: Heavier tails than normal (outliers)
    - S-shaped pattern: Skewed residuals
    - Systematic curvature: Non-normal distribution
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ode_framework.diagnostics.visualizations import plot_qq
    >>> residuals = np.random.normal(0, 1, 100)
    >>> ax = plot_qq(residuals)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to 1D array
    residuals = np.asarray(residuals).flatten()
    
    # Create Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax)
    
    # Formatting
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_variance_trend(
    time: np.ndarray,
    residuals: np.ndarray,
    window: int = 10,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot rolling variance trend to visualize heteroscedasticity.
    
    Computes rolling standard deviation and plots with trend line.
    Useful for detecting if residual variance changes over time.
    
    Parameters
    ----------
    time : np.ndarray
        Time points. Shape: (n_samples,)
    residuals : np.ndarray
        Residuals from ODE solver. Shape: (n_samples,) or (n_samples, 1)
    window : int, optional
        Rolling window size. Default is 10.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes object for further customization
    
    Notes
    -----
    Heteroscedasticity (variance depends on time/state):
    - Trend line slope is significantly non-zero
    - Rolling variance exhibits pattern
    - Suggests model needs state-dependent error model
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ode_framework.diagnostics.visualizations import plot_variance_trend
    >>> time = np.linspace(0, 10, 100)
    >>> residuals = np.random.normal(0, 1, 100)
    >>> ax = plot_variance_trend(time, residuals, window=10)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Convert to arrays
    residuals = np.asarray(residuals).flatten()
    time = np.asarray(time).flatten()
    
    # Ensure window is valid
    window = min(window, len(residuals) // 2)
    if window < 2:
        window = 2
    
    # Compute rolling standard deviation
    rolling_std = np.array([
        np.std(residuals[max(0, i-window):i+window]) 
        for i in range(len(residuals))
    ])
    
    # Plot rolling variance
    ax.plot(time, rolling_std, 'o-', alpha=0.6, linewidth=1.5, markersize=4, 
            color='#2ca02c', label=f'Rolling Std (window={window})')
    
    # Add trend line
    if len(time) > 2:
        coeffs = np.polyfit(time, rolling_std, 1)
        trend_line = np.polyval(coeffs, time)
        ax.plot(time, trend_line, 'r--', linewidth=2, label='Trend')
    
    # Formatting
    ax.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rolling Std Dev', fontsize=11, fontweight='bold')
    ax.set_title('Variance Trend Analysis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    return ax


def plot_diagnostics(
    time: np.ndarray,
    residuals: np.ndarray,
    test_results: dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive 2x2 diagnostic plot with all key visualizations.
    
    Combines all diagnostic plots into a single figure with subplots:
    - Top-left: Residual time series
    - Top-right: Autocorrelation function
    - Bottom-left: Q-Q plot
    - Bottom-right: Variance trend
    
    Parameters
    ----------
    time : np.ndarray
        Time points. Shape: (n_samples,)
    residuals : np.ndarray
        Residuals from ODE solver. Shape: (n_samples,) or (n_samples, 1)
    test_results : dict
        Results from diagnostic tests (for title information).
        Expected keys: 'heteroscedasticity', 'autocorrelation', 'nonstationarity'
    save_path : str, optional
        Path to save figure. If provided, saves figure to this location.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing all subplots
    
    Notes
    -----
    The overall title includes a summary of detected failures from test_results.
    Failure counts and types are extracted and displayed in the main title.
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ode_framework.diagnostics.visualizations import plot_diagnostics
    >>> from ode_framework.diagnostics.diagnostic_engine import DiagnosticEngine
    >>> 
    >>> # Generate test data
    >>> time = np.linspace(0, 10, 100)
    >>> residuals = np.random.normal(0, 1, 100)
    >>> 
    >>> # Run diagnostics
    >>> engine = DiagnosticEngine()
    >>> test_results = engine.run_diagnostics(residuals, time)
    >>> 
    >>> # Create visualization
    >>> fig = plot_diagnostics(time, residuals, test_results)
    >>> plt.show()
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Extract summary information from test_results
    summary = test_results.get('summary', {})
    num_failures = summary.get('num_failures', 0)
    failure_types = summary.get('failure_types', [])
    
    # Create main title with failure information
    if num_failures == 0:
        title_suffix = "✓ All Tests Passed"
        title_color = 'green'
    elif num_failures == 1:
        title_suffix = f"⚠️ 1 Failure: {failure_types[0]}"
        title_color = 'orange'
    else:
        title_suffix = f"⚠️ {num_failures} Failures: {', '.join(failure_types)}"
        title_color = 'red'
    
    main_title = f"ODE Solver Diagnostics - {title_suffix}"
    fig.suptitle(main_title, fontsize=14, fontweight='bold', color=title_color, y=0.995)
    
    # Plot 1: Residual time series (top-left)
    plot_residuals_timeseries(time, residuals, ax=axes[0, 0])
    
    # Plot 2: ACF (top-right)
    plot_acf(residuals, lags=20, ax=axes[0, 1])
    
    # Plot 3: Q-Q plot (bottom-left)
    plot_qq(residuals, ax=axes[1, 0])
    
    # Plot 4: Variance trend (bottom-right)
    plot_variance_trend(time, residuals, window=10, ax=axes[1, 1])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    return fig

