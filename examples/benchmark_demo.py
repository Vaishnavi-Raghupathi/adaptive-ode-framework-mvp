#!/usr/bin/env python
"""Benchmarking infrastructure demonstration.

This script demonstrates the benchmark suite for testing the adaptive ODE
solver framework on standard problems with known characteristics.

The benchmarks include:
1. Van der Pol Oscillator - Periodic nonlinear dynamics
2. Lorenz System - Chaotic dynamics
3. Noisy Exponential Decay - Heteroscedastic noise

Execution
---------
Run from the adaptive-ode-framework-mvp directory:

    python examples/benchmark_demo.py

Expected Output
---------------
- Summary of each benchmark problem
- Data generation examples
- Diagnostic expected patterns
- Full benchmark suite results
- Success metrics and solver selection statistics

Author: Vaishnavi Raghupathi
Date: February 2026
License: Apache 2.0
"""

import numpy as np
import sys
from datetime import datetime

from ode_framework.benchmarks import (
    VanDerPolOscillator,
    LorenzSystem,
    NoisyExponentialDecay,
    run_benchmark_suite
)
from ode_framework.adaptive import AdaptiveSolverFramework


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 75
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    width = 75
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def demo_van_der_pol() -> None:
    """Demonstrate Van der Pol oscillator benchmark."""
    print_header("VAN DER POL OSCILLATOR")
    
    print("""
The Van der Pol oscillator is a nonlinear second-order ODE that exhibits
periodic limit cycle behavior. It's commonly used to test solvers on
nonlinear periodic dynamics.

ODE: d²x/dt² - μ(1-x²)dx/dt + x = 0
Rewritten: dx/dt = y, dy/dt = μ(1-x²)y - x

Characteristics:
- Nonlinear periodic dynamics
- Limit cycle behavior
- Excellent test for autocorrelation detection
- Parameter μ controls nonlinearity strength
""")
    
    # Create oscillator
    oscillator = VanDerPolOscillator(mu=1.0)
    print(f"\nCreated: {oscillator.name} (μ={oscillator.mu})")
    print(f"Description: {oscillator.description}")
    
    # Generate data
    print("\nGenerating data...")
    data = oscillator.generate_data(n_points=200, noise_level=0.05)
    print(f"  Time range: [{data['t'][0]:.1f}, {data['t'][-1]:.1f}]")
    print(f"  Data shape: {data['x'].shape}")
    print(f"  Noise level: {data['noise_level']}")
    
    # Show expected diagnostics
    print("\nExpected diagnostic patterns:")
    expected = oscillator.expected_diagnostics
    for diag_type, expected_val in expected.items():
        print(f"  - {diag_type}: {expected_val}")
    
    # Show data statistics
    print("\nData statistics:")
    print(f"  X range: [{data['x'][:, 0].min():.3f}, {data['x'][:, 0].max():.3f}]")
    print(f"  X std: {data['x'][:, 0].std():.3f}")


def demo_lorenz_system() -> None:
    """Demonstrate Lorenz system benchmark."""
    print_header("LORENZ SYSTEM")
    
    print("""
The Lorenz system is a classic chaotic dynamical system exhibiting sensitive
dependence on initial conditions and the famous butterfly attractor shape.
It's an excellent test for solver robustness on chaotic dynamics.

ODE:
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

Standard parameters (chaos regime):
  σ = 10 (Prandtl number)
  ρ = 28 (Rayleigh number)
  β = 8/3 (geometric factor)

Characteristics:
- Chaotic dynamics (sensitive to initial conditions)
- Butterfly attractor
- Nonlinear coupling between all variables
- Good benchmark for long-term prediction
""")
    
    # Create system
    lorenz = LorenzSystem()
    print(f"\nCreated: {lorenz.name}")
    print(f"Description: {lorenz.description}")
    print(f"Parameters: σ={lorenz.sigma}, ρ={lorenz.rho}, β={lorenz.beta:.3f}")
    
    # Generate data
    print("\nGenerating data...")
    data = lorenz.generate_data(n_points=300, noise_level=0.01)
    print(f"  Time range: [{data['t'][0]:.1f}, {data['t'][-1]:.1f}]")
    print(f"  Data shape: {data['x'].shape}")
    print(f"  Noise level: {data['noise_level']}")
    
    # Show expected diagnostics
    print("\nExpected diagnostic patterns:")
    expected = lorenz.expected_diagnostics
    for diag_type, expected_val in expected.items():
        print(f"  - {diag_type}: {expected_val}")
    
    # Show data statistics
    print("\nData statistics:")
    x_ranges = [
        (data['x'][:, i].min(), data['x'][:, i].max())
        for i in range(data['x'].shape[1])
    ]
    for i, (x_min, x_max) in enumerate(x_ranges):
        print(f"  X{i} range: [{x_min:.3f}, {x_max:.3f}]")


def demo_noisy_exponential_decay() -> None:
    """Demonstrate noisy exponential decay benchmark."""
    print_header("NOISY EXPONENTIAL DECAY")
    
    print("""
A simple linear ODE (exponential decay) with state-dependent heteroscedastic
noise. This is an excellent test for heteroscedasticity detection since the
noise variance scales with the state value.

ODE: dx/dt = -λx
Analytical solution: x(t) = x₀ exp(-λt)

Noise model (heteroscedastic):
  x_observed = x_true + ε
  ε ~ N(0, σ²x_true²)

This creates proportional noise where noise scales with state value.

Characteristics:
- Linear ODE (analytically solvable)
- State-dependent heteroscedastic noise
- Excellent for testing variance detection
- Noise decreases over time as state decays
""")
    
    # Create decay problem
    decay = NoisyExponentialDecay(lambda_decay=0.5, noise_coefficient=0.1)
    print(f"\nCreated: {decay.name}")
    print(f"Description: {decay.description}")
    
    # Generate data
    print("\nGenerating data...")
    data = decay.generate_data(n_points=100)
    print(f"  Time range: [{data['t'][0]:.1f}, {data['t'][-1]:.1f}]")
    print(f"  Data shape: {data['x'].shape}")
    
    # Show expected diagnostics
    print("\nExpected diagnostic patterns:")
    expected = decay.expected_diagnostics
    for diag_type, expected_val in expected.items():
        print(f"  - {diag_type}: {expected_val}")
    
    # Show analytical solution vs noisy
    print("\nAnalytical solution verification:")
    t_test = np.array([0, 1, 2, 5, 10])
    x_analytical = decay.analytical_solution(t_test, x0=data['x0'])
    print(f"  t values: {t_test}")
    print(f"  x analytical: {x_analytical}")
    
    # Show heteroscedasticity
    print("\nHeteroscedasticity (noise ~ state²):")
    noise = data['x'] - data['x_clean']
    for i in [0, 25, 50, 99]:
        noise_std = np.abs(decay.noise_coefficient * data['x_clean'][i, 0])
        noise_actual = np.abs(noise[i, 0])
        print(f"  t={data['t'][i]:.1f}: x={data['x_clean'][i, 0]:.4f}, "
              f"noise_std={noise_std:.4f}, noise={noise_actual:.4f}")


def demo_benchmark_suite() -> None:
    """Run full benchmark suite and show results."""
    print_header("FULL BENCHMARK SUITE")
    
    print("""
Running the adaptive solver framework on all three benchmark problems.
The suite will:
1. Generate synthetic data for each benchmark
2. Fit the adaptive framework
3. Check if diagnostics are detected correctly
4. Track solver selection
5. Compute success statistics
""")
    
    # Create benchmarks
    benchmarks = [
        VanDerPolOscillator(mu=1.0),
        LorenzSystem(),
        NoisyExponentialDecay(noise_coefficient=0.1)
    ]
    
    # Create framework
    print("\nInitializing adaptive framework...")
    framework = AdaptiveSolverFramework(verbose=False)
    
    # Run benchmarks
    print("\nRunning benchmark suite...")
    start_time = datetime.now()
    results = run_benchmark_suite(framework, benchmarks=benchmarks, verbose=True)
    elapsed = datetime.now() - start_time
    
    # Display results
    print_subheader("BENCHMARK RESULTS")
    
    print(f"\nTotal time: {elapsed.total_seconds():.2f}s")
    print(f"\nSummary statistics:")
    print(f"  Total benchmarks: {results['summary']['total_benchmarks']}")
    print(f"  Diagnostic checks: {results['summary']['total_diagnostic_checks']}")
    print(f"  Successful checks: {results['summary']['successful_diagnostics']}")
    print(f"  Success rate: {results['summary']['success_rate']:.1%}")
    
    print(f"\nSolver selections:")
    for solver, count in sorted(
        results['summary']['solver_selections'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {solver}: {count} time(s)")
    
    print_subheader("DETAILED RESULTS")
    for i, result in enumerate(results['results'], 1):
        print(f"\n[{i}] {result['benchmark']}")
        print(f"    Selected solver: {result['solver_selected']}")
        print(f"    Confidence: {result['confidence']}")
        print(f"    Diagnostics detected: {result['diagnostics_detected']}/{result['diagnostics_checked']}")
        
        print(f"\n    Expected diagnostics:")
        for diag_type, expected_val in result['expected_diagnostics'].items():
            print(f"      {diag_type}: {expected_val}")
        
        print(f"\n    Detected diagnostics:")
        for diag_type, detected_val in result['detected_diagnostics'].items():
            match = "✓" if detected_val == result['expected_diagnostics'].get(diag_type) else "✗"
            print(f"      {diag_type}: {detected_val} {match}")


def main() -> None:
    """Run all demonstrations."""
    print_header("BENCHMARK INFRASTRUCTURE DEMONSTRATION")
    
    print("""
This demonstration showcases the benchmarking infrastructure for the
adaptive ODE solver framework. We'll show:

1. Van der Pol Oscillator - Nonlinear periodic dynamics
2. Lorenz System - Chaotic dynamics
3. Noisy Exponential Decay - Heteroscedastic noise
4. Full benchmark suite - Testing on all three

Each benchmark is designed to test specific aspects of the adaptive
framework and diagnostic detection capabilities.
""")
    
    # Run demos
    demo_van_der_pol()
    demo_lorenz_system()
    demo_noisy_exponential_decay()
    demo_benchmark_suite()
    
    # Summary
    print_header("SUMMARY")
    print("""
The benchmarking infrastructure provides:

✓ Three standard ODE test problems with well-known characteristics
✓ Synthetic data generation with configurable noise levels
✓ Expected diagnostic patterns for validation
✓ Comprehensive benchmark suite for evaluating the framework
✓ Success metrics and detailed result analysis

This allows systematic evaluation of the adaptive framework's ability
to detect diagnostic issues and select appropriate solvers across
different classes of ODE systems.

For more information, see:
- ode_framework/benchmarks/benchmark_problems.py
- examples/benchmark_demo.py (this file)
""")
    
    print("\n✓ Demonstration complete!")


if __name__ == "__main__":
    main()
