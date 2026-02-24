"""Benchmark problems for testing ODE solvers and adaptive framework.

This module provides standard benchmark problems used to evaluate solver
performance and test the adaptive framework's ability to detect and handle
various types of ODE system dynamics.

Each benchmark problem includes:
- A well-defined ODE system (often with known solutions)
- Data generation with optional noise
- Expected diagnostic failure patterns
- Performance characteristics

The benchmarks are designed to test:
1. Simple linear dynamics (exponential decay)
2. Periodic dynamics (Van der Pol oscillator)
3. Chaotic dynamics (Lorenz system)
4. Noisy data with heteroscedastic errors

Usage
-----
>>> from ode_framework.benchmarks import VanDerPolOscillator
>>> oscillator = VanDerPolOscillator(mu=1.0)
>>> t, x = oscillator.generate_data(n_points=200)
>>> print(oscillator.expected_diagnostics)

Running full benchmark suite:
>>> from ode_framework.benchmarks import run_benchmark_suite
>>> results = run_benchmark_suite(framework)
"""

from typing import Dict, Tuple, Any, Optional, List
import numpy as np
from scipy.integrate import solve_ivp
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)


class VanDerPolOscillator:
    """Van der Pol oscillator benchmark problem.
    
    The Van der Pol oscillator is a nonlinear second-order ODE that exhibits
    periodic limit cycle behavior. It's commonly used to test solvers on
    nonlinear periodic dynamics.
    
    Mathematical Form
    -----------------
    The original second-order ODE:
        d²x/dt² - μ(1-x²)dx/dt + x = 0
    
    Rewritten as a first-order system:
        dx/dt = y
        dy/dt = μ(1-x²)y - x
    
    where μ > 0 is the nonlinearity parameter.
    
    Characteristics
    ---------------
    - Nonlinear periodic dynamics
    - Limit cycle behavior (converges to periodic orbit)
    - Good test for detecting autocorrelation
    - Parameter μ controls nonlinearity strength
    - No steady state (except for μ=0)
    
    Diagnostic Expectations
    -----------------------
    - Autocorrelation: Expected (periodic dynamics)
    - Heteroscedasticity: No (regular noise propagation)
    - Nonstationarity: No (limit cycle is stationary)
    - State Dependence: Possible (nonlinear coupling)
    
    Parameters
    ----------
    mu : float, default=1.0
        Nonlinearity parameter. Must be > 0.
        - μ < 0.5: Weak nonlinearity, nearly sinusoidal
        - μ = 1.0: Standard parameter
        - μ > 2.0: Strong nonlinearity, sharp corners in limit cycle
    """
    
    def __init__(self, mu: float = 1.0):
        """Initialize Van der Pol oscillator.
        
        Parameters
        ----------
        mu : float, default=1.0
            Nonlinearity parameter (must be > 0).
        
        Raises
        ------
        ValueError
            If mu <= 0.
        """
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        self.mu = mu
        self.name = "Van der Pol Oscillator"
        self.description = f"Van der Pol oscillator with μ={mu}"
    
    def _ode_system(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for Van der Pol oscillator.
        
        Parameters
        ----------
        t : float
            Time (unused, system is autonomous).
        
        state : np.ndarray
            State vector [x, y] where x is position and y is velocity.
        
        Returns
        -------
        np.ndarray
            Time derivatives [dx/dt, dy/dt].
        """
        x, y = state
        dx_dt = y
        dy_dt = self.mu * (1 - x**2) * y - x
        return np.array([dx_dt, dy_dt])
    
    def generate_data(
        self,
        t_span: Tuple[float, float] = (0, 20),
        n_points: int = 200,
        noise_level: float = 0.0,
        initial_conditions: Tuple[float, float] = (2.0, 0.0),
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Generate data from Van der Pol oscillator.
        
        Parameters
        ----------
        t_span : tuple, default=(0, 20)
            Time interval (t_start, t_end) for integration.
        
        n_points : int, default=200
            Number of time points to generate.
        
        noise_level : float, default=0.0
            Standard deviation of Gaussian noise added to solution.
            Set to 0 for noise-free data.
        
        initial_conditions : tuple, default=(2.0, 0.0)
            Initial state (x0, y0).
        
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 't': Time points, shape (n_points,)
            - 'x': State values with noise, shape (n_points, 2)
            - 'x_clean': State values without noise, shape (n_points, 2)
            - 'noise_level': Applied noise level
            - 'mu': Parameter used
        
        Notes
        -----
        The solution is obtained by integrating the ODE system using
        scipy.integrate.solve_ivp with the RK45 method (adaptive solver).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create time points
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Solve ODE with dense output for guaranteed n_points
        solution = solve_ivp(
            self._ode_system,
            t_span,
            initial_conditions,
            method='RK45',
            dense_output=True
        )
        
        # Evaluate at requested time points using dense output
        x_clean = solution.sol(t_eval).T  # Shape: (n_points, 2)
        
        # Add noise if requested
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, x_clean.shape)
            x_noisy = x_clean + noise
        else:
            x_noisy = x_clean.copy()
        
        return {
            't': t_eval,
            'x': x_noisy,
            'x_clean': x_clean,
            'noise_level': noise_level,
            'mu': self.mu,
            'system_name': self.name
        }
    
    @property
    def expected_diagnostics(self) -> Dict[str, bool]:
        """Expected diagnostic failure patterns.
        
        Returns
        -------
        dict
            Dictionary indicating expected diagnostic failures:
            - 'autocorrelated': True (periodic dynamics)
            - 'heteroscedastic': False (regular noise)
            - 'nonstationary': False (limit cycle is stationary)
            - 'state_dependent': Possible (nonlinear coupling)
        """
        return {
            'autocorrelated': True,
            'heteroscedastic': False,
            'nonstationary': False,
            'state_dependent': False
        }


class LorenzSystem:
    """Lorenz system benchmark problem.
    
    The Lorenz system is a classic chaotic dynamical system exhibiting
    sensitive dependence on initial conditions and fractal structure.
    It's an excellent test for solver robustness on chaotic dynamics.
    
    Mathematical Form
    -----------------
    The system of ODEs:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    Standard Parameters (chaos regime):
        σ = 10 (Prandtl number)
        ρ = 28 (Rayleigh number)
        β = 8/3
    
    Characteristics
    ---------------
    - Chaotic dynamics (sensitive to initial conditions)
    - Butterfly attractor (distinctive shape)
    - Nonlinear coupling between all variables
    - Useful for testing long-term prediction
    - Good benchmark for numerical stability
    
    Diagnostic Expectations
    -----------------------
    - Autocorrelation: Expected (complex dynamics)
    - Heteroscedasticity: Possible (dynamics-dependent variance)
    - Nonstationarity: No (attractor is stationary)
    - State Dependence: Likely (nonlinear coupling)
    
    Parameters
    ----------
    sigma : float, default=10.0
        Prandtl number (heat diffusion coefficient).
    
    rho : float, default=28.0
        Rayleigh number (temperature difference).
        - ρ < 1: Fixed point at origin
        - 1 < ρ < ~24.7: Single fixed point
        - ρ > ~24.7: Chaos (butterfly attractor)
    
    beta : float, default=8/3
        Geometric factor (aspect ratio).
    """
    
    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8/3
    ):
        """Initialize Lorenz system.
        
        Parameters
        ----------
        sigma : float, default=10.0
            Prandtl number (must be > 0).
        
        rho : float, default=28.0
            Rayleigh number (must be > 0).
        
        beta : float, default=8/3
            Geometric factor (must be > 0).
        
        Raises
        ------
        ValueError
            If any parameter is <= 0.
        """
        if sigma <= 0 or rho <= 0 or beta <= 0:
            raise ValueError("All parameters must be positive")
        
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.name = "Lorenz System"
        self.description = f"Lorenz system (σ={sigma}, ρ={rho}, β={beta:.3f})"
    
    def _ode_system(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for Lorenz system.
        
        Parameters
        ----------
        t : float
            Time (unused, system is autonomous).
        
        state : np.ndarray
            State vector [x, y, z].
        
        Returns
        -------
        np.ndarray
            Time derivatives [dx/dt, dy/dt, dz/dt].
        """
        x, y, z = state
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def generate_data(
        self,
        t_span: Tuple[float, float] = (0, 50),
        n_points: int = 500,
        noise_level: float = 0.0,
        initial_conditions: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Generate data from Lorenz system.
        
        Parameters
        ----------
        t_span : tuple, default=(0, 50)
            Time interval (t_start, t_end) for integration.
        
        n_points : int, default=500
            Number of time points to generate.
        
        noise_level : float, default=0.0
            Standard deviation of Gaussian noise added to solution.
        
        initial_conditions : tuple, default=(1.0, 1.0, 1.0)
            Initial state (x0, y0, z0).
        
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 't': Time points, shape (n_points,)
            - 'x': State values with noise, shape (n_points, 3)
            - 'x_clean': State values without noise, shape (n_points, 3)
            - 'noise_level': Applied noise level
            - System parameters (sigma, rho, beta)
        
        Notes
        -----
        For long integration times, the chaotic nature means small noise
        differences lead to large divergence. Use noise_level carefully
        when comparing solutions.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create time points
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Solve ODE with dense output for guaranteed n_points
        solution = solve_ivp(
            self._ode_system,
            t_span,
            initial_conditions,
            method='RK45',
            dense_output=True,
            max_step=0.1  # Smaller steps for chaotic system
        )
        
        # Evaluate at requested time points using dense output
        x_clean = solution.sol(t_eval).T  # Shape: (n_points, 3)
        
        # Add noise if requested
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, x_clean.shape)
            x_noisy = x_clean + noise
        else:
            x_noisy = x_clean.copy()
        
        return {
            't': t_eval,
            'x': x_noisy,
            'x_clean': x_clean,
            'noise_level': noise_level,
            'sigma': self.sigma,
            'rho': self.rho,
            'beta': self.beta,
            'system_name': self.name
        }
    
    @property
    def expected_diagnostics(self) -> Dict[str, bool]:
        """Expected diagnostic failure patterns.
        
        Returns
        -------
        dict
            Dictionary indicating expected diagnostic failures:
            - 'autocorrelated': True (complex chaotic dynamics)
            - 'heteroscedastic': Possible (nonlinear variance)
            - 'nonstationary': False (attractor is stationary)
            - 'state_dependent': True (strong nonlinear coupling)
        """
        return {
            'autocorrelated': True,
            'heteroscedastic': False,  # Pure dynamics, no model error
            'nonstationary': False,
            'state_dependent': True  # Nonlinear coupling
        }


class NoisyExponentialDecay:
    """Noisy exponential decay with heteroscedastic errors.
    
    This benchmark tests a simple linear ODE (exponential decay) with
    state-dependent (heteroscedastic) noise. The noise variance scales
    with the current state value, making it an excellent test for
    heteroscedasticity detection.
    
    Mathematical Form
    -----------------
    ODE system:
        dx/dt = -λx
    
    Analytical solution:
        x(t) = x₀ exp(-λt)
    
    Noise model (heteroscedastic):
        x_observed(t) = x_true(t) + ε(t)
        ε(t) ~ N(0, σ²x_true(t)²)
    
    This creates proportional noise: noise scales with the state value.
    
    Characteristics
    ---------------
    - Linear ODE (analytically solvable)
    - State-dependent heteroscedastic noise
    - Excellent for testing variance detection
    - Noise decreases over time (as state decays)
    - Good validation problem for diagnostics
    
    Diagnostic Expectations
    -----------------------
    - Autocorrelation: No (white noise, no model errors)
    - Heteroscedasticity: Yes (noise ~ x²)
    - Nonstationarity: No (exponential decay is stationary)
    - State Dependence: Yes (noise depends on x)
    
    Parameters
    ----------
    lambda_decay : float, default=0.5
        Decay rate. Must be > 0.
        - Higher λ: Faster decay
        - Lower λ: Slower decay
    
    noise_coefficient : float, default=0.1
        Noise proportionality constant σ.
        Defines: noise ~ N(0, σ²x²)
    """
    
    def __init__(
        self,
        lambda_decay: float = 0.5,
        noise_coefficient: float = 0.1
    ):
        """Initialize noisy exponential decay benchmark.
        
        Parameters
        ----------
        lambda_decay : float, default=0.5
            Decay rate (must be > 0).
        
        noise_coefficient : float, default=0.1
            Noise proportionality constant (must be >= 0).
        
        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        if lambda_decay <= 0:
            raise ValueError(f"lambda_decay must be positive, got {lambda_decay}")
        if noise_coefficient < 0:
            raise ValueError(f"noise_coefficient must be non-negative, got {noise_coefficient}")
        
        self.lambda_decay = lambda_decay
        self.noise_coefficient = noise_coefficient
        self.name = "Noisy Exponential Decay"
        self.description = (
            f"Exponential decay with heteroscedastic noise "
            f"(λ={lambda_decay}, σ={noise_coefficient})"
        )
    
    def _ode_system(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for exponential decay.
        
        Parameters
        ----------
        t : float
            Time (unused).
        
        state : np.ndarray
            State value [x].
        
        Returns
        -------
        np.ndarray
            Time derivative [dx/dt].
        """
        x = state[0]
        dx_dt = -self.lambda_decay * x
        return np.array([dx_dt])
    
    def analytical_solution(self, t: np.ndarray, x0: float = 1.0) -> np.ndarray:
        """Compute analytical solution.
        
        Parameters
        ----------
        t : np.ndarray
            Time points.
        
        x0 : float, default=1.0
            Initial condition.
        
        Returns
        -------
        np.ndarray
            Analytical solution x(t) = x₀ exp(-λt).
        """
        return x0 * np.exp(-self.lambda_decay * t)
    
    def generate_data(
        self,
        t_span: Tuple[float, float] = (0, 10),
        n_points: int = 100,
        initial_condition: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Generate data from exponential decay with heteroscedastic noise.
        
        Parameters
        ----------
        t_span : tuple, default=(0, 10)
            Time interval (t_start, t_end).
        
        n_points : int, default=100
            Number of time points.
        
        initial_condition : float, default=1.0
            Initial state value x₀.
        
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 't': Time points, shape (n_points,)
            - 'x': Noisy observations, shape (n_points, 1)
            - 'x_clean': Analytical solution, shape (n_points, 1)
            - 'noise_level': Heteroscedastic noise coefficient
            - 'lambda_decay': Decay rate used
            - 'x0': Initial condition
        
        Notes
        -----
        The noise is heteroscedastic (state-dependent):
            noise_std = σ * |x_true|
        
        This creates realistic measurement errors where larger values
        have proportionally larger absolute errors.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create time points
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        # Compute analytical solution
        x_clean = self.analytical_solution(t, initial_condition).reshape(-1, 1)
        
        # Add heteroscedastic noise: ε ~ N(0, (σ*x)²)
        if self.noise_coefficient > 0:
            # Noise standard deviation scales with state value
            noise_std = self.noise_coefficient * np.abs(x_clean)
            noise = np.random.normal(0, noise_std)
            x_noisy = x_clean + noise
        else:
            x_noisy = x_clean.copy()
        
        return {
            't': t,
            'x': x_noisy,
            'x_clean': x_clean,
            'noise_level': self.noise_coefficient,
            'lambda_decay': self.lambda_decay,
            'x0': initial_condition,
            'system_name': self.name
        }
    
    @property
    def expected_diagnostics(self) -> Dict[str, bool]:
        """Expected diagnostic failure patterns.
        
        Returns
        -------
        dict
            Dictionary indicating expected diagnostic failures:
            - 'autocorrelated': False (white noise)
            - 'heteroscedastic': True (noise ∝ x²)
            - 'nonstationary': False (exponential decay is deterministic)
            - 'state_dependent': True (noise depends on x)
        """
        return {
            'autocorrelated': False,
            'heteroscedastic': True,  # Noise proportional to x²
            'nonstationary': False,
            'state_dependent': True   # Noise depends on state
        }


def run_benchmark_suite(
    framework,
    benchmarks: Optional[list] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run full benchmark suite on adaptive framework.
    
    This function runs the framework on all benchmark problems and collects
    statistics about solver selection and diagnostic detection.
    
    Parameters
    ----------
    framework : AdaptiveSolverFramework
        Fitted adaptive framework to test.
    
    benchmarks : list, optional
        List of benchmark instances to run. If None, uses defaults.
    
    verbose : bool, default=True
        Print progress information.
    
    Returns
    -------
    dict
        Benchmark results including:
        - 'results': List of results for each benchmark
        - 'summary': Aggregated statistics
        - 'success_rate': Fraction of correct diagnostic detections
    
    Examples
    --------
    >>> from ode_framework.adaptive import AdaptiveSolverFramework
    >>> from ode_framework.benchmarks import run_benchmark_suite
    >>> framework = AdaptiveSolverFramework()
    >>> results = run_benchmark_suite(framework)
    >>> print(f"Success rate: {results['summary']['success_rate']:.1%}")
    """
    if benchmarks is None:
        benchmarks = [
            VanDerPolOscillator(mu=1.0),
            LorenzSystem(),
            NoisyExponentialDecay(noise_coefficient=0.1)
        ]
    
    results = {
        'results': [],
        'summary': {
            'total_benchmarks': len(benchmarks),
            'successful_diagnostics': 0,
            'total_diagnostic_checks': 0,
            'solver_selections': {},
            'diagnostic_detections': {}
        }
    }
    
    for i, benchmark in enumerate(benchmarks, 1):
        if verbose:
            print(f"\n[{i}/{len(benchmarks)}] Running {benchmark.name}...")
        
        # Generate data
        # Note: NoisyExponentialDecay doesn't take noise_level parameter
        # (it's set via noise_coefficient at initialization)
        if isinstance(benchmark, NoisyExponentialDecay):
            data = benchmark.generate_data()
        else:
            data = benchmark.generate_data(noise_level=0.05)
        
        # For multivariate systems, use first component for diagnostics
        # (diagnostic engine expects 1D residuals)
        x_for_fit = data['x'][:, [0]] if data['x'].ndim > 1 else data['x']
        
        # Fit framework
        framework.reset()
        framework.fit(data['t'], x_for_fit)
        
        # Get results
        report = framework.get_selection_report()
        expected = benchmark.expected_diagnostics
        
        # Check diagnostic detections
        diagnostics = report['diagnostics']
        detected_correctly = 0
        checked = 0
        
        for diag_type, expected_val in expected.items():
            checked += 1
            results['summary']['total_diagnostic_checks'] += 1
            
            if diag_type == 'autocorrelated':
                detected = diagnostics['autocorrelated']
            elif diag_type == 'heteroscedastic':
                detected = diagnostics['heteroscedastic']
            elif diag_type == 'nonstationary':
                detected = diagnostics['nonstationary']
            else:
                continue
            
            if detected == expected_val:
                detected_correctly += 1
                results['summary']['successful_diagnostics'] += 1
        
        # Track solver selection
        solver_name = report['selected_method']
        if solver_name not in results['summary']['solver_selections']:
            results['summary']['solver_selections'][solver_name] = 0
        results['summary']['solver_selections'][solver_name] += 1
        
        # Store result
        result = {
            'benchmark': benchmark.name,
            'solver_selected': solver_name,
            'confidence': report['confidence'],
            'diagnostics_detected': detected_correctly,
            'diagnostics_checked': checked,
            'expected_diagnostics': expected,
            'detected_diagnostics': {
                'autocorrelated': diagnostics['autocorrelated'],
                'heteroscedastic': diagnostics['heteroscedastic'],
                'nonstationary': diagnostics['nonstationary']
            }
        }
        results['results'].append(result)
        
        if verbose:
            print(
                f"  → Selected: {solver_name} (confidence: {report['confidence']})"
            )
            print(
                f"  → Diagnostics: {detected_correctly}/{checked} correct"
            )
    
    # Calculate success rate
    total_checks = results['summary']['total_diagnostic_checks']
    if total_checks > 0:
        results['summary']['success_rate'] = (
            results['summary']['successful_diagnostics'] / total_checks
        )
    else:
        results['summary']['success_rate'] = 0.0
    
    return results


def check_expected_diagnostics(
    actual_diagnostics: Dict[str, Any],
    expected: Dict[str, bool]
) -> bool:
    """Check if diagnostic results match expected patterns.
    
    Parameters
    ----------
    actual_diagnostics : dict
        Actual diagnostic results from framework fitting.
    
    expected : dict
        Expected diagnostic patterns (True/False for each test).
    
    Returns
    -------
    bool
        True if all expected patterns match actual results.
    
    Notes
    -----
    Extracts boolean test results from nested diagnostic dictionaries
    and compares against expected patterns.
    """
    checks_passed = 0
    checks_total = 0
    
    for diag_type, expected_value in expected.items():
        checks_total += 1
        
        if diag_type == 'autocorrelated':
            actual = actual_diagnostics.get('autocorrelation', {}).get('autocorrelated', False)
        elif diag_type == 'heteroscedastic':
            actual = actual_diagnostics.get('heteroscedasticity', {}).get('heteroscedastic', False)
        elif diag_type == 'nonstationary':
            actual = actual_diagnostics.get('nonstationarity', {}).get('nonstationary', False)
        elif diag_type == 'state_dependent':
            actual = actual_diagnostics.get('state_dependence', {}).get('state_dependent', False) if actual_diagnostics.get('state_dependence') else False
        else:
            continue
        
        if actual == expected_value:
            checks_passed += 1
    
    # All checks must pass
    return checks_passed == checks_total if checks_total > 0 else False


def compute_all_metrics(
    x_actual: np.ndarray,
    x_predicted: np.ndarray
) -> Dict[str, float]:
    """Compute performance metrics for solver evaluation.
    
    Parameters
    ----------
    x_actual : np.ndarray
        Actual state values.
    
    x_predicted : np.ndarray
        Predicted state values.
    
    Returns
    -------
    dict
        Dictionary with metrics:
        - 'rmse': Root mean squared error
        - 'mse': Mean squared error
        - 'mae': Mean absolute error
        - 'r_squared': R² coefficient of determination
        - 'nrmse': Normalized RMSE
    """
    # Flatten for univariate and multivariate cases
    y_true = x_actual.flatten()
    y_pred = x_predicted.flatten()
    
    # Mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² coefficient
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Normalized RMSE (by actual range)
    y_range = np.ptp(y_true)
    nrmse = (rmse / y_range) if y_range > 0 else np.inf
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'nrmse': nrmse
    }


def run_comprehensive_benchmarks(
    framework,
    problems: Optional[List] = None,
    noise_levels: Optional[List[float]] = None,
    n_trials: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """Run comprehensive benchmark evaluation with varying difficulty.
    
    This function tests the framework across multiple problems and noise levels,
    running multiple trials for statistical robustness. For each configuration,
    it measures:
    - Prediction accuracy (RMSE, R²)
    - Computational performance (fit/predict time)
    - Diagnostic detection accuracy
    - Solver selection consistency
    
    Parameters
    ----------
    framework : AdaptiveSolverFramework
        Framework to benchmark.
    
    problems : list, optional
        List of benchmark problem instances. If None, uses defaults:
        [VanDerPolOscillator(), LorenzSystem(), NoisyExponentialDecay()]
    
    noise_levels : list, optional
        List of noise levels to test. Default: [0.0, 0.02, 0.05]
    
    n_trials : int, default=3
        Number of trials per configuration for averaging.
    
    verbose : bool, default=True
        Print progress information.
    
    Returns
    -------
    pd.DataFrame
        Results table with columns:
        - 'problem': Problem name
        - 'noise_level': Applied noise level
        - 'rmse_mean': Mean RMSE across trials
        - 'rmse_std': Std dev of RMSE
        - 'mae_mean': Mean absolute error
        - 'r2_mean': Mean R² score
        - 'r2_std': Std dev of R²
        - 'diagnostics_correct': All diagnostics detected correctly
        - 'solver_used': Selected solver name
        - 'fit_time_mean': Mean fitting time (seconds)
        - 'predict_time_mean': Mean prediction time (seconds)
        - 'n_trials': Number of trials (for reference)
    
    Examples
    --------
    >>> from ode_framework.benchmarks import run_comprehensive_benchmarks
    >>> from ode_framework.adaptive import AdaptiveSolverFramework
    >>> framework = AdaptiveSolverFramework()
    >>> results = run_comprehensive_benchmarks(
    ...     framework,
    ...     noise_levels=[0.0, 0.05],
    ...     n_trials=5
    ... )
    >>> print(results.to_string())
    >>> # Save results
    >>> results.to_csv('benchmark_results.csv', index=False)
    
    Notes
    -----
    This is a more thorough evaluation than run_benchmark_suite(), which
    provides quick validation. Use this for comprehensive performance analysis.
    """
    if problems is None:
        problems = [
            VanDerPolOscillator(mu=1.0),
            LorenzSystem(),
            NoisyExponentialDecay(noise_coefficient=0.1)
        ]
    
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05]
    
    results = []
    total_configs = len(problems) * len(noise_levels)
    current_config = 0
    
    for problem in problems:
        for noise_level in noise_levels:
            current_config += 1
            
            if verbose:
                print(
                    f"\n[{current_config}/{total_configs}] "
                    f"{problem.name} (noise={noise_level:.3f})"
                )
            
            trial_metrics = []
            
            for trial in range(n_trials):
                if verbose:
                    print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)
                
                # Generate data
                if isinstance(problem, NoisyExponentialDecay):
                    data = problem.generate_data()
                else:
                    data = problem.generate_data(noise_level=noise_level)
                
                # Use first component for univariate diagnostics
                x_for_fit = data['x'][:, [0]] if data['x'].ndim > 1 else data['x']
                t = data['t']
                
                # Fit framework
                start_time = time.time()
                framework.reset()
                framework.fit(t, x_for_fit)
                fit_time = time.time() - start_time
                
                # Predict
                start_time = time.time()
                x_pred = framework.predict(t)
                pred_time = time.time() - start_time
                
                # Compute metrics
                metrics = compute_all_metrics(x_for_fit, x_pred)
                
                # Check diagnostics
                diagnostics_correct = False
                if framework.diagnostic_history:
                    diag = framework.diagnostic_history[-1]
                    diagnostics_correct = check_expected_diagnostics(
                        diag,
                        problem.expected_diagnostics
                    )
                
                trial_metrics.append({
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r_squared'],
                    'fit_time': fit_time,
                    'pred_time': pred_time,
                    'diag_correct': diagnostics_correct,
                    'solver': framework.selected_solver_name
                })
                
                if verbose:
                    print(f"RMSE={metrics['rmse']:.4f}, R²={metrics['r_squared']:.4f}")
            
            # Aggregate trials
            rmse_values = [m['rmse'] for m in trial_metrics]
            mae_values = [m['mae'] for m in trial_metrics]
            r2_values = [m['r2'] for m in trial_metrics]
            fit_times = [m['fit_time'] for m in trial_metrics]
            pred_times = [m['pred_time'] for m in trial_metrics]
            
            result = {
                'problem': problem.name,
                'noise_level': noise_level,
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'mae_mean': np.mean(mae_values),
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values),
                'diagnostics_correct': all([m['diag_correct'] for m in trial_metrics]),
                'solver_used': trial_metrics[0]['solver'],
                'fit_time_mean': np.mean(fit_times),
                'predict_time_mean': np.mean(pred_times),
                'n_trials': n_trials
            }
            results.append(result)
    
    return pd.DataFrame(results)
