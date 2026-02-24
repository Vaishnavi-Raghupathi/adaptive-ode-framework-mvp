"""Decision module for automatic solver selection.

This module provides rule-based decision logic to automatically recommend
the most appropriate ODE solver based on diagnostic test results from
the Week 2 diagnostics module.

The key idea is that different failure patterns in diagnostic tests
(heteroscedasticity, autocorrelation, nonstationarity, state-dependence)
suggest different solver approaches:

- No failures: Classical solver (RK4, RK45) is sufficient
- Heteroscedasticity: Stochastic differential equations (SDE)
- Autocorrelation: Neural ODEs or regime-switching
- Nonstationarity: Time-varying parameters or regime-switching
- Multiple failures: Ensemble or Neural ODE approaches

This systematizes expert knowledge about solver selection into
transparent, interpretable decision rules.

Classes
-------
MethodSelector
    Main decision engine mapping diagnostic patterns to solver methods.

Functions
---------
recommend_method
    Quick function to get method recommendation from diagnostic results.

Example
-------
>>> from ode_framework.decision import MethodSelector, recommend_method
>>> from ode_framework.diagnostics import DiagnosticEngine
>>> 
>>> # Get diagnostic results
>>> engine = DiagnosticEngine()
>>> results = engine.run_diagnostics(residuals, t_data)
>>> 
>>> # Method 1: Using MethodSelector directly
>>> selector = MethodSelector(prefer_speed=True)
>>> recommendation = selector.select_method(results)
>>> 
>>> # Method 2: Quick recommendation
>>> recommendation = recommend_method(results)
>>> 
>>> print(f"Recommended method: {recommendation['method']}")
>>> print(f"Confidence: {recommendation['confidence']}")
>>> print(f"Reasoning: {recommendation['reasoning']}")
"""

from .method_selector import MethodSelector, recommend_method

__all__ = [
    'MethodSelector',
    'recommend_method',
]
