"""Benchmarking module for ODE solver framework.

This module provides standard benchmark problems for evaluating solver
performance and testing the adaptive framework's diagnostic capabilities.

Classes
-------
VanDerPolOscillator
    Nonlinear periodic oscillator with limit cycle behavior.
    Good for testing autocorrelation detection.

LorenzSystem
    Chaotic dynamical system with butterfly attractor.
    Tests robustness on nonlinear coupled equations.

NoisyExponentialDecay
    Simple linear ODE with state-dependent heteroscedastic noise.
    Perfect for testing variance heterogeneity detection.

Functions
---------
run_benchmark_suite
    Run all benchmarks on an adaptive framework and collect statistics.

Examples
--------
Basic usage:

>>> from ode_framework.benchmarks import VanDerPolOscillator
>>> oscillator = VanDerPolOscillator(mu=1.0)
>>> data = oscillator.generate_data(n_points=200, noise_level=0.05)
>>> print(data['x'].shape)
(200, 2)

Running full benchmark suite:

>>> from ode_framework.benchmarks import run_benchmark_suite
>>> from ode_framework.adaptive import AdaptiveSolverFramework
>>> framework = AdaptiveSolverFramework()
>>> results = run_benchmark_suite(framework)
>>> print(f"Success rate: {results['summary']['success_rate']:.1%}")

Accessing expected diagnostics:

>>> oscillator = VanDerPolOscillator()
>>> print(oscillator.expected_diagnostics)
{'autocorrelated': True, 'heteroscedastic': False, ...}
"""

from .benchmark_problems import (
    VanDerPolOscillator,
    LorenzSystem,
    NoisyExponentialDecay,
    run_benchmark_suite
)

__all__ = [
    'VanDerPolOscillator',
    'LorenzSystem',
    'NoisyExponentialDecay',
    'run_benchmark_suite'
]
