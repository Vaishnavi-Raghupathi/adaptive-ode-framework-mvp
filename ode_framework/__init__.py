"""Adaptive ODE Framework - Week 1 MVP and Beyond.

A comprehensive Python framework for solving Ordinary Differential Equations (ODEs)
with automatic system identification using machine learning. This package combines
SINDy (Sparse Identification of Nonlinear Dynamics) with advanced numerical
integration methods to discover and solve differential equations from data.

Core Modules:
    - solvers: ODE solver implementations (RK4, RK45)
    - metrics: Error metrics and performance evaluation
    - utils: Test problems and utility functions
    - diagnostics: Solver validation and statistical tests (Week 2)
    - decision: Automatic solver selection based on diagnostics (Week 3)
    - adaptive: Complete adaptive pipeline orchestration (Week 3+)

Quick Start:
    >>> from ode_framework.solvers.classical import RK45Solver
    >>> from ode_framework.utils.test_problems import exponential_decay
    >>> import numpy as np
    >>> 
    >>> # Generate test data
    >>> t = np.linspace(0, 5, 50)
    >>> problem = exponential_decay(t, noise_level=0.01)
    >>> 
    >>> # Create and fit solver
    >>> solver = RK45Solver()
    >>> solver.fit(problem['t'], problem['x_exact'])
    >>> 
    >>> # Make predictions
    >>> t_new = np.linspace(0, 5, 100)
    >>> x_pred = solver.predict(t_new)
    >>> 
    >>> # Evaluate accuracy
    >>> from ode_framework.metrics.error_metrics import rmse
    >>> error = rmse(problem['x_exact'], x_pred[:len(problem['x_exact'])])
    >>> print(f"RMSE: {error:.4f}")

Subpackages:
    - solvers: RK4Solver, RK45Solver, BaseSolver (ABC)
    - metrics: L2 norm, MSE, RMSE, R-squared, compute_all_metrics
    - utils: exponential_decay, harmonic_oscillator, logistic_growth
    - diagnostics: Statistical tests, DiagnosticEngine, visualizations
    - decision: MethodSelector, automatic solver recommendation
    - adaptive: AdaptiveSolverFramework, complete pipeline

Version: 0.3.0 (Week 3 Adaptive Pipeline)
Author: Vaishnavi Raghupathi
License: Apache 2.0
"""

__version__ = "0.3.0"
__author__ = "Vaishnavi Raghupathi"
__email__ = "vaishnavi.raghupathi@example.com"

# Core imports - expose main API
from . import solvers
from . import metrics
from . import utils
from . import diagnostics
from . import decision
from . import adaptive

__all__ = [
    "solvers",
    "metrics",
    "utils",
    "diagnostics",
    "decision",
    "adaptive",
    "__version__",
]
