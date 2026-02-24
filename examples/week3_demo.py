#!/usr/bin/env python
"""Week 3 Decision Framework - Automatic Solver Selection Demo.

This script demonstrates the Week 3 decision module, which uses diagnostic
results from Week 2 to automatically recommend the most appropriate ODE solver.

The demonstration shows:
1. Generating synthetic ODE data with various challenges
2. Running diagnostic tests (Week 2)
3. Getting automatic solver recommendations (Week 3)
4. Generating comprehensive decision reports
5. Analyzing confidence levels and alternatives

Execution
---------
Run from the adaptive-ode-framework-mvp directory:

    python examples/week3_demo.py

Expected Output
---------------
- Console output showing diagnostic results and recommendations
- Decision reports with confidence levels
- Comparison of methods for different data scenarios
- Alternative solver suggestions

Author: Vaishnavi Raghupathi
Date: February 2026
License: Apache 2.0
"""

import numpy as np
from datetime import datetime
import sys

# Week 1 modules
from ode_framework.solvers.classical import RK45Solver, RK4Solver
from ode_framework.utils.test_problems import exponential_decay, logistic_growth

# Week 2 modules
from ode_framework.diagnostics import DiagnosticEngine

# Week 3 modules
from ode_framework.decision import MethodSelector, recommend_method


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    width = 70
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def scenario_1_clean_data() -> None:
    """Scenario 1: Clean data with no issues.
    
    This is the ideal case where a classical solver works well.
    Diagnosis should show no failures → Classical solver recommended.
    """
    print_header("SCENARIO 1: CLEAN DATA")
    print("Data: Exponential decay with minimal noise (1%)")
    print("Expected: No diagnostic failures → Classical solver")
    
    # Generate clean data
    t = np.linspace(0, 5, 100)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5, noise_level=0.01)
    t_data = problem['t']
    x_data = problem['x_exact']
    
    # Fit solver
    print("\n[1] Fitting RK45Solver...")
    solver = RK45Solver()
    solver.fit(t_data, x_data)
    residuals = solver.compute_residuals(t_data, x_data)
    print(f"    ✓ Residuals std: {np.std(residuals):.6f}")
    
    # Run diagnostics
    print("\n[2] Running diagnostics...")
    engine = DiagnosticEngine(verbose=False)
    results = engine.run_diagnostics(residuals, t_data)
    
    # Get recommendation
    print("\n[3] Getting solver recommendation...")
    recommendation = recommend_method(results)
    print(f"    Recommended: {recommendation['method']}")
    print(f"    Confidence: {recommendation['confidence']}")
    print(f"    Reasoning: {recommendation['reasoning']}")
    
    # Generate detailed report
    print("\n[4] Generating decision report...")
    selector = MethodSelector()
    report = selector.generate_decision_report(results, recommendation)
    
    print(f"\n    Decision Path: {report['decision_tree_path'][0]}")
    print(f"    P-value Summary:")
    p_analysis = report['p_value_analysis']
    for test_name in ['heteroscedasticity', 'autocorrelation', 'nonstationarity']:
        p_val = p_analysis[test_name]['p_value']
        detected = p_analysis[test_name]['detected']
        status = "✗ FAILED" if detected else "✓ PASSED"
        print(f"      {test_name:20} p={p_val:.4f}  {status}")


def scenario_2_heteroscedastic_data() -> None:
    """Scenario 2: Data with heteroscedasticity.
    
    Variance changes over time - typical of systems with noise
    proportional to state magnitude.
    Diagnosis should detect heteroscedasticity → SDE recommended.
    """
    print_header("SCENARIO 2: HETEROSCEDASTIC DATA")
    print("Data: Exponential decay with state-dependent noise")
    print("Expected: Heteroscedasticity detected → SDE solver")
    
    # Generate data with heteroscedastic noise
    t = np.linspace(0, 5, 100)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5, noise_level=0.0)
    x_true = problem['x_exact']
    
    # Add heteroscedastic noise (proportional to magnitude)
    np.random.seed(42)
    noise = np.random.normal(0, 0.1 * np.abs(x_true), x_true.shape)
    x_noisy = x_true + noise
    
    # Fit solver
    print("\n[1] Fitting RK45Solver to noisy data...")
    solver = RK45Solver()
    solver.fit(t, x_noisy)
    residuals = solver.compute_residuals(t, x_noisy)
    print(f"    ✓ Residuals std: {np.std(residuals):.6f}")
    
    # Run diagnostics
    print("\n[2] Running diagnostics...")
    engine = DiagnosticEngine(verbose=False)
    results = engine.run_diagnostics(residuals, t)
    
    # Get recommendation
    print("\n[3] Getting solver recommendation...")
    recommendation = recommend_method(results)
    print(f"    Recommended: {recommendation['method']}")
    print(f"    Confidence: {recommendation['confidence']}")
    print(f"    Reasoning: {recommendation['reasoning']}")
    
    # Show alternatives
    print("\n[4] Alternative solvers:")
    selector = MethodSelector()
    alternatives = selector.get_alternative_solvers(recommendation['method'])
    for i, alt in enumerate(alternatives[:3], 1):
        print(f"      {i}. {alt}")
    
    # Generate detailed report
    print("\n[5] Confidence breakdown:")
    report = selector.generate_decision_report(results, recommendation)
    cb = report['confidence']['breakdown']
    print(f"    Evidence Quality: {cb.get('evidence_quality', 'N/A')}")
    print(f"    Method Fit: {cb.get('method_fit', 'N/A')}")
    print(f"    Recommendation: {cb.get('recommendation', 'N/A')}")


def scenario_3_autocorrelated_data() -> None:
    """Scenario 3: Data with autocorrelated residuals.
    
    Residuals are correlated - suggests missed dynamics or
    incorrect model structure.
    Diagnosis should detect autocorrelation → Neural ODE recommended.
    """
    print_header("SCENARIO 3: AUTOCORRELATED RESIDUALS")
    print("Data: Exponential decay with systematic model mismatch")
    print("Expected: Autocorrelation detected → Neural ODE solver")
    
    # Generate data with systematic bias
    t = np.linspace(0, 5, 100)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5, noise_level=0.0)
    x_true = problem['x_exact']
    
    # Add systematic pattern (model mismatch)
    systematic_error = 0.05 * np.sin(2 * np.pi * t / 5).reshape(-1, 1)
    noise = np.random.normal(0, 0.02, x_true.shape)
    x_noisy = x_true + systematic_error + noise
    
    # Fit solver
    print("\n[1] Fitting RK45Solver with model mismatch...")
    solver = RK45Solver()
    solver.fit(t, x_noisy)
    residuals = solver.compute_residuals(t, x_noisy)
    print(f"    ✓ Residuals std: {np.std(residuals):.6f}")
    
    # Run diagnostics
    print("\n[2] Running diagnostics...")
    engine = DiagnosticEngine(verbose=False)
    results = engine.run_diagnostics(residuals, t)
    
    # Get recommendation
    print("\n[3] Getting solver recommendation...")
    recommendation = recommend_method(results)
    print(f"    Recommended: {recommendation['method']}")
    print(f"    Confidence: {recommendation['confidence']}")
    
    # Show test results
    print("\n[4] Diagnostic test results:")
    selector = MethodSelector()
    report = selector.generate_decision_report(results, recommendation)
    
    p_analysis = report['p_value_analysis']
    for test_name in ['heteroscedasticity', 'autocorrelation', 'nonstationarity']:
        p_val = p_analysis[test_name]['p_value']
        detected = p_analysis[test_name]['detected']
        status = "✗ FAILED" if detected else "✓ PASSED"
        print(f"    {test_name:20} p={p_val:.4f}  {status}")
    
    print(f"\n    Next Steps:")
    for i, step in enumerate(recommendation['next_steps'][:3], 1):
        print(f"      {i}. {step}")


def scenario_4_multiple_failures() -> None:
    """Scenario 4: Complex data with multiple diagnostic failures.
    
    Multiple issues: heteroscedasticity, autocorrelation, nonstationarity.
    Diagnosis should recommend the most sophisticated method.
    """
    print_header("SCENARIO 4: COMPLEX DATA WITH MULTIPLE FAILURES")
    print("Data: Exponential decay with noise, trend, and missing dynamics")
    print("Expected: Multiple failures detected → Ensemble or Neural")
    
    # Generate data with multiple issues
    t = np.linspace(0, 5, 150)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5, noise_level=0.0)
    x_true = problem['x_exact']
    
    # Add multiple failure modes
    # 1. Heteroscedastic noise
    hetero_noise = np.random.normal(0, 0.1 * np.abs(x_true), x_true.shape)
    
    # 2. Systematic error (autocorrelation)
    systematic_error = 0.05 * np.sin(2 * np.pi * t / 5).reshape(-1, 1)
    
    # 3. Time-varying drift (nonstationarity)
    trend = 0.02 * (t / t[-1]).reshape(-1, 1)
    
    x_noisy = x_true + hetero_noise + systematic_error + trend
    
    # Fit solver
    print("\n[1] Fitting RK45Solver to complex data...")
    solver = RK45Solver()
    solver.fit(t, x_noisy)
    residuals = solver.compute_residuals(t, x_noisy)
    print(f"    ✓ Residuals std: {np.std(residuals):.6f}")
    
    # Run diagnostics
    print("\n[2] Running diagnostics...")
    engine = DiagnosticEngine(verbose=False)
    results = engine.run_diagnostics(residuals, t)
    
    # Get recommendation
    print("\n[3] Getting solver recommendation...")
    recommendation = recommend_method(results)
    print(f"    Recommended: {recommendation['method']}")
    print(f"    Confidence: {recommendation['confidence']}")
    
    # Detailed analysis
    print("\n[4] Failure Analysis:")
    selector = MethodSelector()
    report = selector.generate_decision_report(results, recommendation)
    
    diag_summary = report['diagnostic_summary']
    print(f"    Num Failures: {diag_summary['num_failures']}")
    print(f"    Failure Types: {', '.join(diag_summary['failure_types'])}")
    
    print(f"\n[5] Confidence Breakdown:")
    cb = report['confidence']['breakdown']
    print(f"    Evidence Quality: {cb.get('evidence_quality', 'N/A')}")
    print(f"    Strong Signals: {cb.get('strong_signals', 0)}")
    print(f"    Moderate Signals: {cb.get('moderate_signals', 0)}")
    
    print(f"\n[6] Alternative Methods:")
    for i, alt in enumerate(recommendation['alternatives'], 1):
        print(f"      {i}. {alt}")


def scenario_5_constrained_selection() -> None:
    """Scenario 5: Method selection with user constraints.
    
    Demonstrates how constraints can guide solver selection
    when computational resources are limited.
    """
    print_header("SCENARIO 5: CONSTRAINED METHOD SELECTION")
    print("Data: Exponential decay with heteroscedasticity")
    print("Constraint: Maximum complexity = 'moderate', prefer speed")
    print("Expected: SDE solver (respects both diagnostic needs and constraints)")
    
    # Generate data with heteroscedasticity
    t = np.linspace(0, 5, 100)
    problem = exponential_decay(t, x0=1.0, lambda_=0.5, noise_level=0.0)
    x_true = problem['x_exact']
    
    # Heteroscedastic noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1 * np.abs(x_true), x_true.shape)
    x_noisy = x_true + noise
    
    # Fit solver
    print("\n[1] Fitting RK45Solver...")
    solver = RK45Solver()
    solver.fit(t, x_noisy)
    residuals = solver.compute_residuals(t, x_noisy)
    
    # Run diagnostics
    print("\n[2] Running diagnostics...")
    engine = DiagnosticEngine(verbose=False)
    results = engine.run_diagnostics(residuals, t)
    
    # Get recommendation WITHOUT constraints
    print("\n[3] Recommendation WITHOUT constraints:")
    selector_unconstrained = MethodSelector()
    rec_unconstrained = selector_unconstrained.select_method(results)
    print(f"    Method: {rec_unconstrained['method']}")
    print(f"    Confidence: {rec_unconstrained['confidence']}")
    
    # Get recommendation WITH constraints
    print("\n[4] Recommendation WITH constraints (max_complexity='moderate'):")
    selector_constrained = MethodSelector(max_complexity='moderate', prefer_speed=True)
    rec_constrained = selector_constrained.select_method(results)
    print(f"    Method: {rec_constrained['method']}")
    print(f"    Confidence: {rec_constrained['confidence']}")
    
    # Show method characteristics
    print("\n[5] Method Characteristics:")
    print(f"\n    Selected Method: {rec_constrained['method']}")
    chars = selector_constrained.METHOD_CHARACTERISTICS[rec_constrained['method']]
    for key, val in chars.items():
        if key != 'suited_for':
            print(f"      {key:18}: {val}")


def main() -> None:
    """Run all demonstration scenarios."""
    print("\n" + "="*70)
    print("WEEK 3: DECISION FRAMEWORK - AUTOMATIC SOLVER SELECTION".center(70))
    print("="*70)
    print("\nThis demonstration shows how Week 3 automatically recommends")
    print("solvers based on diagnostic results from Week 2.")
    print("\nExecuting 5 scenarios with different data characteristics...")
    
    try:
        # Run all scenarios
        scenario_1_clean_data()
        scenario_2_heteroscedastic_data()
        scenario_3_autocorrelated_data()
        scenario_4_multiple_failures()
        scenario_5_constrained_selection()
        
        print_header("DEMONSTRATION COMPLETE")
        print("\n✅ All scenarios executed successfully!")
        print("\nKey Takeaways:")
        print("  1. Diagnostics (Week 2) identify failure patterns")
        print("  2. Decision logic (Week 3) recommends appropriate methods")
        print("  3. Confidence levels indicate recommendation certainty")
        print("  4. Alternative methods provide fallback options")
        print("  5. User constraints can guide solver selection")
        print("\nFor more information, see:")
        print("  - WEEK2_TESTING_SUMMARY.md (diagnostic tests)")
        print("  - examples/week2_demo.py (diagnostic demonstration)")
        print("  - examples/week3_demo.py (this file)")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
