"""Diagnostics module for ODE solver validation and analysis.

This module provides comprehensive diagnostics tools for validating ODE solver
fits and predictions. It includes:

    - Statistical Tests: Residual analysis, autocorrelation, stationarity
    - Diagnostic Engine: Orchestrates multiple tests and generates reports
    - Visualizations: Diagnostic plots and analysis figures

Quick Start:
    >>> from ode_framework.diagnostics import DiagnosticEngine, plot_diagnostics
    >>> from ode_framework.solvers.classical import RK45Solver
    >>> 
    >>> # Create and fit solver
    >>> solver = RK45Solver()
    >>> solver.fit(t_data, x_data)
    >>> 
    >>> # Run diagnostics
    >>> engine = DiagnosticEngine(solver)
    >>> report = engine.run_diagnostics(t_data, x_data)
    >>> print(report.summary())
    >>> 
    >>> # Visualize diagnostics
    >>> plot_diagnostics(solver, t_data, x_data)

Submodules:
    - statistical_tests: Breusch-Pagan, Ljung-Box, ADF tests
    - diagnostic_engine: DiagnosticEngine and DiagnosticReport classes
    - visualizations: Plotting functions for diagnostics
"""

from .statistical_tests import (
    breusch_pagan_test,
    ljung_box_test,
    augmented_dickey_fuller_test,
    state_dependence_test,
)
from .diagnostic_engine import (
    DiagnosticEngine,
    DiagnosticReport,
)
from .visualizations import (
    plot_diagnostics,
    plot_residuals_timeseries,
    plot_acf,
    plot_qq,
    plot_variance_trend,
)

__all__ = [
    # Statistical tests
    "breusch_pagan_test",
    "ljung_box_test",
    "augmented_dickey_fuller_test",
    "state_dependence_test",
    # Diagnostic engine
    "DiagnosticEngine",
    "DiagnosticReport",
    # Visualizations
    "plot_diagnostics",
    "plot_residuals_timeseries",
    "plot_acf",
    "plot_qq",
    "plot_variance_trend",
]
