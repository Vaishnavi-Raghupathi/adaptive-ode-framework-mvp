"""Adaptive ODE solver framework with automatic method selection.

This module provides the complete adaptive pipeline that automatically
selects the best solver based on data characteristics.

Classes
-------
AdaptiveSolverFramework
    Main orchestrator for adaptive solver selection.

Example
-------
>>> from ode_framework.adaptive import AdaptiveSolverFramework
>>> framework = AdaptiveSolverFramework()
>>> framework.fit(t_data, x_data)
>>> predictions = framework.predict(t_eval)
>>> report = framework.get_selection_report()
"""

from .adaptive_framework import AdaptiveSolverFramework

__all__ = ['AdaptiveSolverFramework']
