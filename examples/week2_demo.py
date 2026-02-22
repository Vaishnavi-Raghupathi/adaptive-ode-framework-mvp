#!/usr/bin/env python3
"""Week 2 Diagnostics Framework Demo.

This script demonstrates the complete Week 2 diagnostics workflow:
1. Generate test ODE data with noise
2. Fit a solver to the noisy data
3. Compute residuals
4. Run comprehensive statistical diagnostics
5. Generate diagnostic visualizations
6. Produce a formatted diagnostic report

Output:
- Diagnostic test results in the console
- Diagnostic plots saved as 'week2_diagnostics.png'
- Formatted summary table
- Recommendations for solver improvements
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Try importing required modules
try:
    from ode_framework.solvers.classical import RK45Solver
    from ode_framework.utils.test_problems import exponential_decay
    from ode_framework.diagnostics.diagnostic_engine import DiagnosticEngine
    from ode_framework.diagnostics.visualizations import plot_diagnostics
except ImportError as e:
    print(f"Error: Missing required module - {e}")
    print("Install with: pip install pysindy statsmodels")
    sys.exit(1)

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pysindy")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, description: str) -> None:
    """Print a formatted step header."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 70)


def main() -> None:
    """Run the Week 2 diagnostics demonstration."""
    print_section("WEEK 2: DIAGNOSTICS FRAMEWORK - FULL DEMONSTRATION")
    print("\nThis demo showcases:")
    print("  • Fitting an ODE solver to noisy data")
    print("  • Computing and analyzing residuals")
    print("  • Running statistical diagnostic tests")
    print("  • Generating diagnostic visualizations")
    print("  • Interpreting diagnostic recommendations")

    start_time = time.time()

    # =========================================================================
    # STEP 1: Generate synthetic ODE data with noise
    # =========================================================================
    print_step(1, "Generate synthetic exponential decay data with noise")

    # Create time grid: 150 points from 0 to 5 seconds
    t_eval = np.linspace(0, 5, 150)
    print(f"  • Time points: {len(t_eval)} (0 to 5 seconds)")

    # Generate exponential decay with 3% measurement noise
    problem = exponential_decay(t_eval, x0=1.0, lambda_=0.5, noise_level=0.03)
    x_noisy = problem["x_exact"]
    x_true = exponential_decay(t_eval, x0=1.0, lambda_=0.5, noise_level=0.0)["x_exact"]

    print(f"  • ODE: dx/dt = -0.5*x, x(0) = 1.0")
    print(f"  • Noise level: 3% (realistic measurement uncertainty)")
    print(f"  • Data shape: {x_noisy.shape}")
    print(f"  • Data range: [{x_noisy.min():.4f}, {x_noisy.max():.4f}]")
    print(f"  • True solution range: [{x_true.min():.4f}, {x_true.max():.4f}]")
    print(f"  ✓ Synthetic data generated successfully")

    # =========================================================================
    # STEP 2: Fit RK45 solver to noisy data
    # =========================================================================
    print_step(2, "Fit RK45Solver to noisy data using SINDy system identification")

    solver = RK45Solver(rtol=1e-6, atol=1e-9)
    print(f"  • Solver: RK45Solver (adaptive Runge-Kutta 4/5)")
    print(f"  • Tolerance: rtol=1e-6, atol=1e-9")
    print(f"  • Training data points: {len(t_eval)}")

    try:
        solver.fit(t_eval, x_noisy)
        print(f"  ✓ Solver fitted successfully using SINDy")
        print(f"  • Learned ODE model from {len(t_eval)} noisy observations")
    except Exception as e:
        print(f"  ✗ Error fitting solver: {e}")
        print("  Note: pysindy may need optimization for this data")
        sys.exit(1)

    # =========================================================================
    # STEP 3: Compute predictions and residuals
    # =========================================================================
    print_step(3, "Generate predictions and compute residuals")

    x_pred = solver.predict(t_eval)
    residuals = x_noisy - x_pred

    print(f"  • Predictions computed on {len(t_eval)} time points")
    print(f"  • Residuals (errors): actual - predicted")
    print(f"  • Residuals shape: {residuals.shape}")
    print(f"  • Residuals statistics:")
    print(f"    - Mean: {np.mean(residuals):>10.6f} (ideally ~0)")
    print(f"    - Std Dev: {np.std(residuals):>10.6f}")
    print(f"    - Min: {np.min(residuals):>10.6f}")
    print(f"    - Max: {np.max(residuals):>10.6f}")
    print(f"    - Range: {np.max(residuals) - np.min(residuals):.6f}")

    # Check if residuals look reasonable
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"  • RMSE: {rmse:.6f}")
    print(f"  ✓ Residuals computed successfully")

    # =========================================================================
    # STEP 4: Run comprehensive diagnostic tests
    # =========================================================================
    print_step(4, "Run comprehensive statistical diagnostic tests")

    print("  • Diagnostic tests evaluate residual quality:")
    print("    - Heteroscedasticity: Does variance change over time?")
    print("    - Autocorrelation: Are residuals correlated?")
    print("    - Stationarity: Is the mean/variance constant?")
    print("    - State-dependence: Do residuals correlate with state?")
    print()

    # Create diagnostic engine and run all tests
    engine = DiagnosticEngine(verbose=False)
    print(f"  Executing diagnostic suite...")
    results = engine.run_diagnostics(residuals, t_eval, state_vars=x_pred)

    print(f"  ✓ All diagnostic tests completed successfully")
    print()

    # Extract and display individual test results
    print("  Test Results:")
    print("  " + "-" * 66)

    tests = {
        "heteroscedasticity": "Breusch-Pagan Test (Heteroscedasticity)",
        "autocorrelation": "Ljung-Box Test (Autocorrelation)",
        "nonstationarity": "ADF Test (Stationarity)",
        "state_dependence": "State-Dependence Test"
    }

    test_details = {}
    for key, name in tests.items():
        result = results[key]
        if result is not None:
            p_value = result.get("p_value", np.nan)
            detected = result.get(list(result.keys())[1], False)  # Get the detection key
            status = "FAILED ✗" if detected else "PASSED ✓"
            test_details[name] = {
                "p_value": p_value,
                "detected": detected,
                "status": status
            }
            print(f"    {name:<50} {status:>8}")
            print(f"      p-value: {p_value:.6f}")
        else:
            print(f"    {name:<50} {'SKIPPED':<8}")

    print("  " + "-" * 66)

    # =========================================================================
    # STEP 5: Generate diagnostic report
    # =========================================================================
    print_step(5, "Generate diagnostic report and recommendations")

    report = engine.generate_report()
    print("\n" + report)

    # Extract summary information
    summary = results["summary"]
    failure_detected = summary["failure_detected"]
    failure_types = summary["failure_types"]
    recommendation = summary["recommended_method"]
    confidence = summary.get("confidence", "N/A")

    print()
    print("  Summary:")
    print(f"    • Failures detected: {'YES' if failure_detected else 'NO'}")
    if failure_detected:
        print(f"    • Failure types: {', '.join(failure_types)}")
    print(f"    • Recommended method: {recommendation}")
    print(f"    • Confidence: {confidence}")

    # =========================================================================
    # STEP 6: Create diagnostic visualizations
    # =========================================================================
    print_step(6, "Create diagnostic visualizations")

    print("  Creating 2x2 diagnostic plot with:")
    print("    • Residuals over time")
    print("    • ACF plot (autocorrelation function)")
    print("    • Distribution histogram")
    print("    • Q-Q plot (normality test)")
    print()

    try:
        # Create figure for diagnostics
        # plot_diagnostics requires: time, residuals, test_results (the full results dict)
        fig = plot_diagnostics(t_eval, residuals, results)

        # Save the plot
        output_dir = Path(__file__).parent
        output_file = output_dir / "week2_diagnostics.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  ✓ Diagnostic plot saved successfully")
        print(f"    • Location: {output_file}")
        print(f"    • File size: {output_file.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"  ⚠ Warning: Could not create visualizations - {e}")
        print(f"    This is optional and doesn't affect analysis")

    # =========================================================================
    # STEP 7: Print summary table
    # =========================================================================
    print_step(7, "Formatted diagnostic summary table")

    print("\n  DIAGNOSTIC TEST RESULTS")
    print("  " + "=" * 66)
    print(f"  {'Test Name':<30} {'Detected':<12} {'P-Value':<12} {'Status':<12}")
    print("  " + "-" * 66)

    for name, details in test_details.items():
        detected_str = "YES" if details["detected"] else "NO"
        p_value_str = f"{details['p_value']:.6f}"
        status_str = "FAILED ✗" if details["detected"] else "PASSED ✓"
        print(f"  {name:<30} {detected_str:<12} {p_value_str:<12} {status_str:<12}")

    print("  " + "=" * 66)

    # =========================================================================
    # STEP 8: Interpretation and next steps
    # =========================================================================
    print_step(8, "Interpretation and recommended next steps")

    print("\n  Current Diagnostics Interpretation:")
    print("  " + "-" * 66)

    if failure_detected:
        print(f"  ⚠ Issues Detected: {', '.join(failure_types)}")
        print()
        print("  This indicates that the current solver may not be capturing")
        print("  all aspects of the system dynamics. Consider:")
        print()

        if "heteroscedastic" in failure_types:
            print("  • Heteroscedasticity detected (variance changes over time)")
            print("    → Increase solver tolerance (tighter integration)")
            print("    → Try different numerical method")
            print("    → Consider stochastic ODE (SDE) formulation")
            print()

        if "autocorrelated" in failure_types:
            print("  • Autocorrelation detected (residuals are correlated)")
            print("    → Increase model complexity or add missing terms")
            print("    → Use machine learning-based identification (e.g., Neural ODE)")
            print("    → Check for unmodeled dynamics")
            print()

        if "nonstationary" in failure_types:
            print("  • Non-stationarity detected (mean/variance changes)")
            print("    → System may have regime changes")
            print("    → Consider switching/regime models")
            print("    → Check for time-varying parameters")
            print()

        if "state_dependent" in failure_types:
            print("  • State-dependence detected (errors depend on state value)")
            print("    → Model residuals as function of state")
            print("    → Use local error modeling")
            print("    → Consider adaptive error correction")
            print()
    else:
        print("  ✓ All diagnostic tests passed!")
        print()
        print("  The solver produces residuals that are:")
        print("    • Homoscedastic (constant variance)")
        print("    • Independent (not autocorrelated)")
        print("    • Stationary (constant mean/variance)")
        print("    • State-independent (errors don't depend on state value)")
        print()
        print("  This indicates a good model fit with minimal structure in errors.")
        print("  The solver is capturing the essential dynamics well.")
        print()

    print("  Recommended Next Steps:")
    print("  " + "-" * 66)
    print(f"  1. Review the diagnostic plot: examples/week2_diagnostics.png")
    print(f"  2. Examine specific failure types above")
    print(f"  3. Consider the recommended solver method: {recommendation}")
    print(f"  4. Re-run diagnostics after solver improvements")
    print()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time

    print_section("DEMONSTRATION COMPLETE")
    print(f"\n  Total execution time: {elapsed:.2f} seconds")
    print(f"  Output files:")
    print(f"    • Diagnostic plot: examples/week2_diagnostics.png")
    print(f"\n  Key Takeaways:")
    print(f"    • Diagnostic framework provides automated model validation")
    print(f"    • Statistical tests detect systematic residual patterns")
    print(f"    • Reports guide solver improvements and method selection")
    print(f"    • Visualizations enable manual inspection of results")
    print()
    print("  For more information, see WEEK2_TESTING_SUMMARY.md")
    print()


if __name__ == "__main__":
    main()
