"""Week 1 demonstration: Full ODE solver pipeline.

This script demonstrates the complete adaptive ODE framework Week 1 pipeline:
1. Generate analytical ODE solutions with noise
2. Fit a solver to noisy data using SINDy
3. Predict on a finer time grid
4. Evaluate accuracy with error metrics
5. Visualize results
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from ode_framework.solvers.classical import RK45Solver
from ode_framework.utils.test_problems import exponential_decay
from ode_framework.metrics.error_metrics import compute_all_metrics


def print_metrics_table(metrics: dict, data_name: str = "Results") -> None:
    """Print metrics in a formatted table.

    Parameters
    ----------
    metrics : dict
        Dictionary containing 'l2_norm', 'mse', 'rmse', 'r_squared'.
    data_name : str, optional
        Name label for the metrics. Default is "Results".
    """
    print("\n" + "=" * 60)
    print(f"{data_name} - Error Metrics")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':<20} {'Description':<20}")
    print("-" * 60)
    print(f"{'L2 Norm':<20} {metrics['l2_norm']:<20.6e} {'Euclidean distance':<20}")
    print(f"{'MSE':<20} {metrics['mse']:<20.6e} {'Mean squared error':<20}")
    print(f"{'RMSE':<20} {metrics['rmse']:<20.6e} {'Root mean squared':<20}")
    print(f"{'R² Score':<20} {metrics['r_squared']:<20.6f} {'Coeff. of determ.':<20}")
    print("=" * 60 + "\n")


def main() -> None:
    """Run the Week 1 demonstration pipeline."""
    print("\n" + "=" * 60)
    print("WEEK 1: Adaptive ODE Framework - Full Pipeline Demo")
    print("=" * 60)

    print("\n[STEP 1] Generate analytical exponential decay data with noise...")
    t_train = np.linspace(0, 5, 100)
    problem = exponential_decay(
        t_train, x0=1.0, lambda_=0.5, noise_level=0.02
    )
    x_noisy = problem["x_exact"]
    x0 = problem["x0"]
    params = problem["params"]

    print(f"  ✓ Generated {len(t_train)} training points")
    print(f"  ✓ Initial condition: x0 = {x0}")
    print(f"  ✓ ODE parameters: λ = {params['lambda']}")
    print(f"  ✓ Data shape: {x_noisy.shape}")

    print("\n[STEP 2] Generate analytical solution on finer grid...")
    t_fine = np.linspace(0, 5, 200)
    problem_fine = exponential_decay(
        t_fine, x0=x0, lambda_=params["lambda"], noise_level=0.0
    )
    x_exact = problem_fine["x_exact"]

    print(f"  ✓ Generated {len(t_fine)} evaluation points")
    print(f"  ✓ Fine grid shape: {x_exact.shape}")

    print("\n[STEP 3] Fit RK45Solver to noisy training data using SINDy...")
    try:
        solver = RK45Solver(rtol=1e-6, atol=1e-9)
        solver.fit(t_train, x_noisy)
        print("  ✓ Solver fitted successfully")
        print(f"  ✓ Learned ODE model from {len(t_train)} noisy observations")
    except ImportError as e:
        print(f"  ✗ Error: {e}")
        print("  Install pysindy: pip install pysindy")
        return

    print("\n[STEP 4] Predict on fine evaluation grid...")
    x_pred_fine = solver.predict(t_fine)
    print(f"  ✓ Predictions shape: {x_pred_fine.shape}")

    print("\n[STEP 5] Compute error metrics on fine grid...")
    metrics_fine = compute_all_metrics(x_exact, x_pred_fine)
    print_metrics_table(metrics_fine, "Fine Grid Evaluation")

    print("\n[STEP 6] Predict on original training times...")
    x_pred_train = solver.predict(t_train)
    residuals = solver.compute_residuals(t_train, x_noisy)
    metrics_train = compute_all_metrics(x_noisy, x_pred_train)
    print_metrics_table(metrics_train, "Training Data Fit")

    print(f"  Mean residual: {np.mean(residuals):.6e}")
    print(f"  Std residual:  {np.std(residuals):.6e}")

    print("\n[STEP 7] Create visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(t_fine, x_exact, "b-", linewidth=2, label="Analytical solution")
    ax1.scatter(t_train, x_noisy, s=20, alpha=0.6, color="red", label="Noisy observations")
    ax1.plot(t_fine, x_pred_fine, "g--", linewidth=2, label="RK45 predictions")
    ax1.set_xlabel("Time (t)", fontsize=11)
    ax1.set_ylabel("State (x)", fontsize=11)
    ax1.set_title("Week 1: ODE Solver Fit to Noisy Data", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    error_fine = x_exact - x_pred_fine.reshape(-1, 1)
    ax2.plot(t_fine, error_fine, "r-", linewidth=2, label="Prediction error")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.fill_between(t_fine, error_fine.flatten(), alpha=0.3)
    ax2.set_xlabel("Time (t)", fontsize=11)
    ax2.set_ylabel("Error (exact - pred)", fontsize=11)
    ax2.set_title(f"Prediction Error (RMSE: {metrics_fine['rmse']:.4e})", 
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / "week1_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Plot saved to: {output_path}")

    print("\n[STEP 8] Summary statistics...")
    print("=" * 60)
    print("WEEK 1 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Training data points:      {len(t_train)}")
    print(f"Evaluation points:         {len(t_fine)}")
    print(f"Noise level (training):    2.0%")
    print(f"ODE system:                Exponential decay (linear)")
    print(f"Solver:                    RK45 (adaptive Runge-Kutta)")
    print(f"System identification:     SINDy (Sparse ID)")
    print("-" * 60)
    print(f"Fine grid RMSE:            {metrics_fine['rmse']:.6e}")
    print(f"Fine grid R²:              {metrics_fine['r_squared']:.6f}")
    print(f"Training fit RMSE:         {metrics_train['rmse']:.6e}")
    print(f"Training fit R²:           {metrics_train['r_squared']:.6f}")
    print("=" * 60)

    print("\n✅ Week 1 pipeline completed successfully!")
    print("   Framework is ready for Week 2 enhancements.\n")


if __name__ == "__main__":
    main()
