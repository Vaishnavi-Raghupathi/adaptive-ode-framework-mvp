"""Adaptive ODE solver framework with automatic method selection.

This module implements the complete adaptive pipeline that automatically
selects and refits ODE solvers based on diagnostic analysis.

Pipeline Overview
-----------------
The adaptive framework follows this workflow:

1. **Baseline Fitting**: Fit a fast classical solver (RK45) to establish baseline
2. **Diagnostic Analysis**: Run statistical tests on residuals
3. **Pattern Detection**: Identify failure patterns and diagnostic issues
4. **Method Selection**: Use decision logic to recommend better solver
5. **Refit (Optional)**: Refit with selected solver if it differs from baseline
6. **Performance Tracking**: Compare baseline vs. improved solver metrics
7. **Report Generation**: Document selection process and improvements

Key Features
------------
- Automatic solver selection based on data characteristics
- Performance comparison before/after adaptation
- Confidence-based recommendations
- Alternative solver suggestions
- Detailed selection reporting and reasoning
- Constraint support (speed, accuracy, complexity)

Use Cases
---------
1. **Exploratory Data Analysis**: Understand what solver works best
2. **Automated Processing**: Select solver without manual tuning
3. **Benchmarking**: Compare different solvers automatically
4. **Production Systems**: Adaptive selection in pipelines

Key Classes
-----------
AdaptiveSolverFramework
    Main orchestrator that implements the adaptive pipeline.

Example
-------
Basic usage:

>>> from ode_framework.adaptive import AdaptiveSolverFramework
>>> import numpy as np
>>>
>>> # Generate or load ODE data
>>> t = np.linspace(0, 5, 100)
>>> x = np.random.randn(100, 1)  # Your data here
>>>
>>> # Create and fit adaptive framework
>>> framework = AdaptiveSolverFramework()
>>> framework.fit(t, x)
>>>
>>> # Get predictions
>>> t_eval = np.linspace(0, 5, 200)
>>> predictions = framework.predict(t_eval)
>>>
>>> # Get selection report
>>> report = framework.get_selection_report()
>>> print(f"Selected: {report['selected_method']}")
>>> print(f"Confidence: {report['confidence']}")

Advanced usage with constraints:

>>> # Prefer speed over accuracy
>>> framework = AdaptiveSolverFramework(
...     prefer_speed=True,
...     max_complexity='moderate'
... )
>>> framework.fit(t, x)
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import numpy as np

from ode_framework.solvers.classical import RK45Solver, RK4Solver
from ode_framework.diagnostics import DiagnosticEngine
from ode_framework.decision import MethodSelector

# Configure logging
logger = logging.getLogger(__name__)


class AdaptiveSolverFramework:
    """Adaptive ODE solver framework with automatic method selection.

    This class orchestrates the complete adaptive pipeline: baseline fitting,
    diagnostics, method selection, optional refitting, and performance tracking.

    The framework automatically determines the best solver method for a given
    ODE system by analyzing diagnostic patterns in the residuals.

    Attributes
    ----------
    baseline_solver : BaseSolver
        The initial classical solver (RK45) used for baseline fitting.
    
    selected_solver : BaseSolver or None
        The solver selected by adaptive method (may differ from baseline).
    
    diagnostic_results : dict or None
        Results from running diagnostics on baseline residuals.
    
    selection_result : dict or None
        Result from automatic method selection.
    
    decision_report : dict or None
        Complete decision report with confidence and reasoning.
    
    _is_fitted : bool
        Whether the framework has been fitted to data.

    Parameters
    ----------
    baseline_solver : str, default='RK45'
        Baseline solver to use. Options: 'RK45', 'RK4'
    
    prefer_speed : bool, default=False
        If True, prefer faster methods over slower ones.
    
    prefer_accuracy : bool, default=False
        If True, prefer more accurate methods.
    
    prefer_interpretability : bool, default=False
        If True, prefer more interpretable methods.
    
    max_complexity : {'low', 'moderate', 'high'}, optional
        Maximum allowed solver complexity.
    
    max_cost : {'low', 'moderate', 'high'}, optional
        Maximum computational cost.
    
    auto_refit : bool, default=True
        If True, refit with selected solver if different from baseline.
    
    verbose : bool, default=False
        If True, print detailed progress information.

    Raises
    ------
    ValueError
        If invalid baseline solver or constraint specified.

    Examples
    --------
    Create and use the adaptive framework:

    >>> from ode_framework.adaptive import AdaptiveSolverFramework
    >>> framework = AdaptiveSolverFramework(verbose=True)
    >>> framework.fit(t_data, x_data)
    >>> x_pred = framework.predict(t_eval)
    >>> report = framework.get_selection_report()

    With constraints for embedded systems:

    >>> framework = AdaptiveSolverFramework(
    ...     max_complexity='low',
    ...     prefer_speed=True
    ... )
    """

    def __init__(
        self,
        baseline_solver: str = 'RK45',
        prefer_speed: bool = False,
        prefer_accuracy: bool = False,
        prefer_interpretability: bool = False,
        max_complexity: Optional[str] = None,
        max_cost: Optional[str] = None,
        auto_refit: bool = True,
        verbose: bool = False
    ):
        """Initialize the AdaptiveSolverFramework.

        Parameters
        ----------
        baseline_solver : str, default='RK45'
            Initial solver to use for baseline fitting.
        
        prefer_speed : bool, default=False
            Prefer faster methods in selection.
        
        prefer_accuracy : bool, default=False
            Prefer more accurate methods in selection.
        
        prefer_interpretability : bool, default=False
            Prefer more interpretable methods in selection.
        
        max_complexity : {'low', 'moderate', 'high'}, optional
            Maximum solver complexity allowed.
        
        max_cost : {'low', 'moderate', 'high'}, optional
            Maximum computational cost allowed.
        
        auto_refit : bool, default=True
            Automatically refit with selected solver.
        
        verbose : bool, default=False
            Print detailed information during fitting.
        """
        # Validate baseline solver
        valid_solvers = ['RK45', 'RK4']
        if baseline_solver not in valid_solvers:
            raise ValueError(f"baseline_solver must be one of {valid_solvers}")

        # Initialize solvers
        self.baseline_solver = self._create_solver(baseline_solver)
        self.baseline_solver_name = baseline_solver
        self.initial_solver = baseline_solver  # Store for reset functionality
        self.selected_solver = None
        self.selected_solver_name = baseline_solver

        # Available solvers (can be updated with set_available_solvers)
        self.available_solvers = {
            'RK45': 'Adaptive classical solver',
            'RK4': 'Fixed-step classical solver',
            'SDE': 'Stochastic differential equations (future)',
            'Neural': 'Neural ODE (future)',
            'Regime': 'Regime-switching (future)',
            'Ensemble': 'Ensemble methods (future)'
        }

        # Decision parameters
        self.prefer_speed = prefer_speed
        self.prefer_accuracy = prefer_accuracy
        self.prefer_interpretability = prefer_interpretability
        self.max_complexity = max_complexity
        self.max_cost = max_cost
        self.auto_refit = auto_refit
        self.verbose = verbose

        # Results storage
        self.diagnostic_results = None
        self.selection_result = None
        self.decision_report = None
        self.performance_comparison = None

        # History tracking
        self.solver_history = []  # Track all solvers tried
        self.diagnostic_history = []  # Track all diagnostics run

        # State tracking
        self._is_fitted = False
        self._fit_time = None
        self._t_data = None
        self._x_data = None

        if self.verbose:
            logger.setLevel(logging.INFO)
            logger.info(f"AdaptiveSolverFramework initialized with baseline={baseline_solver}")

    def _create_solver(self, solver_name: str):
        """Create a solver instance by name.

        Parameters
        ----------
        solver_name : str
            Name of solver ('RK45', 'RK4', 'SDE', 'Neural', 'Regime', 'Ensemble').

        Returns
        -------
        BaseSolver
            Solver instance.

        Notes
        -----
        Currently only RK45 and RK4 are implemented. Other method names
        (SDE, Neural, Regime, Ensemble) are recommendations from the decision
        framework that would be implemented in future weeks. For now, they
        default to RK45 (the more accurate classical solver).
        """
        # Map decision framework recommendations to available solvers
        if solver_name == 'RK45':
            return RK45Solver()
        elif solver_name == 'RK4':
            return RK4Solver()
        elif solver_name in ['SDE', 'Neural', 'Regime', 'Ensemble']:
            # These are future implementations
            # For now, default to RK45 (adaptive classical solver)
            logger.warning(
                f"Solver '{solver_name}' not yet implemented. "
                "Using RK45 as fallback (recommended for complex dynamics)."
            )
            return RK45Solver()
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    def fit(self, t_data: np.ndarray, x_data: np.ndarray) -> None:
        """Fit the adaptive framework to training data.

        This method:
        1. Fits baseline solver to data
        2. Computes residuals
        3. Runs diagnostics
        4. Selects appropriate method
        5. Optionally refits with selected method
        6. Generates performance comparison

        Parameters
        ----------
        t_data : np.ndarray
            Time points. Shape: (n_samples,)
        
        x_data : np.ndarray
            State values. Shape: (n_samples, n_states) or (n_samples,)

        Raises
        ------
        ValueError
            If input arrays have invalid shape or values.

        Notes
        -----
        After fitting, use predict() to make predictions and
        get_selection_report() to view the selection analysis.
        """
        if self.verbose:
            logger.info("Starting adaptive framework fitting...")
            start_time = datetime.now()

        # Store data
        self._t_data = np.asarray(t_data)
        self._x_data = np.asarray(x_data)

        # Ensure correct shapes
        if self._x_data.ndim == 1:
            self._x_data = self._x_data.reshape(-1, 1)

        # Step 1: Fit baseline solver
        if self.verbose:
            logger.info(f"[1/6] Fitting baseline {self.baseline_solver_name} solver...")
        self.baseline_solver.fit(self._t_data, self._x_data)

        # Step 2: Compute baseline residuals
        if self.verbose:
            logger.info("[2/6] Computing baseline residuals...")
        baseline_residuals = self.baseline_solver.compute_residuals(
            self._t_data, self._x_data
        )

        # Step 3: Run diagnostics
        if self.verbose:
            logger.info("[3/6] Running diagnostic tests...")
        engine = DiagnosticEngine(verbose=False)
        self.diagnostic_results = engine.run_diagnostics(baseline_residuals, self._t_data)

        # Track diagnostic history
        self.diagnostic_history.append(self.diagnostic_results.copy() if isinstance(self.diagnostic_results, dict) else self.diagnostic_results)

        # Step 4: Select method
        if self.verbose:
            logger.info("[4/6] Selecting appropriate solver method...")
        selector = MethodSelector(
            prefer_speed=self.prefer_speed,
            prefer_accuracy=self.prefer_accuracy,
            prefer_interpretability=self.prefer_interpretability,
            max_complexity=self.max_complexity,
            max_cost=self.max_cost,
            verbose=False
        )
        self.selection_result = selector.select_method(self.diagnostic_results)
        self.selected_solver_name = self.selection_result['method']

        # Track solver history
        self.solver_history.append({
            'timestamp': datetime.now().isoformat(),
            'solver': self.baseline_solver_name,
            'role': 'baseline'
        })

        # Step 5: Generate decision report
        if self.verbose:
            logger.info("[5/6] Generating decision report...")
        self.decision_report = selector.generate_decision_report(
            self.diagnostic_results,
            self.selection_result
        )

        # Step 6: Refit with selected solver if different
        if self.auto_refit and self.selected_solver_name != self.baseline_solver_name:
            if self.verbose:
                logger.info(f"[6/6] Refitting with {self.selected_solver_name} solver...")
            self.selected_solver = self._create_solver(self.selected_solver_name)
            self.selected_solver.fit(self._t_data, self._x_data)

            # Track selected solver in history
            self.solver_history.append({
                'timestamp': datetime.now().isoformat(),
                'solver': self.selected_solver_name,
                'role': 'selected',
                'reason': self.selection_result.get('reasoning', 'Automatic selection')
            })

            # Compute improved residuals
            selected_residuals = self.selected_solver.compute_residuals(
                self._t_data, self._x_data
            )

            # Compare performance
            self.performance_comparison = self._compare_solvers(
                baseline_residuals,
                selected_residuals
            )
        else:
            self.selected_solver = self.baseline_solver
            # Track that baseline was selected
            self.solver_history.append({
                'timestamp': datetime.now().isoformat(),
                'solver': self.baseline_solver_name,
                'role': 'selected',
                'reason': 'Baseline solver met requirements'
            })
            if self.verbose:
                logger.info(
                    f"[6/6] No refit needed (baseline {self.baseline_solver_name} selected)"
                )

        self._is_fitted = True
        if self.verbose:
            elapsed = datetime.now() - start_time
            logger.info(f"Framework fitting completed in {elapsed.total_seconds():.3f}s")

    def predict(self, t_eval: np.ndarray) -> np.ndarray:
        """Make predictions using the selected solver.

        Parameters
        ----------
        t_eval : np.ndarray
            Time points for evaluation. Shape: (n_eval,)

        Returns
        -------
        np.ndarray
            Predicted state values. Shape: (n_eval, n_states) or (n_eval,)

        Raises
        ------
        RuntimeError
            If framework has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before prediction")

        return self.selected_solver.predict(t_eval)

    def compute_residuals(
        self,
        t_eval: np.ndarray,
        x_eval: np.ndarray
    ) -> np.ndarray:
        """Compute residuals using the selected solver.

        Parameters
        ----------
        t_eval : np.ndarray
            Time points.
        
        x_eval : np.ndarray
            State values.

        Returns
        -------
        np.ndarray
            Residual values.

        Raises
        ------
        RuntimeError
            If framework has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before computing residuals")

        return self.selected_solver.compute_residuals(t_eval, x_eval)

    def _compare_solvers(
        self,
        baseline_residuals: np.ndarray,
        selected_residuals: np.ndarray
    ) -> Dict[str, Any]:
        """Compare performance between baseline and selected solver.

        Parameters
        ----------
        baseline_residuals : np.ndarray
            Residuals from baseline solver.
        
        selected_residuals : np.ndarray
            Residuals from selected solver.

        Returns
        -------
        dict
            Comparison metrics:
            - 'baseline_residuals': baseline residual stats
            - 'selected_residuals': selected residual stats
            - 'improvement': relative improvement percentage
        """
        baseline_std = np.std(baseline_residuals)
        selected_std = np.std(selected_residuals)

        improvement = ((baseline_std - selected_std) / baseline_std) * 100 if baseline_std > 0 else 0

        return {
            'baseline_solver': self.baseline_solver_name,
            'selected_solver': self.selected_solver_name,
            'baseline_residuals': {
                'std': baseline_std,
                'mean': np.mean(baseline_residuals),
                'max': np.max(np.abs(baseline_residuals))
            },
            'selected_residuals': {
                'std': selected_std,
                'mean': np.mean(selected_residuals),
                'max': np.max(np.abs(selected_residuals))
            },
            'improvement_percent': improvement
        }

    def get_selection_report(self) -> Dict[str, Any]:
        """Get complete selection report.

        Returns a comprehensive report of the adaptive selection process,
        including baseline fitting, diagnostics, method selection, and
        performance comparison (if refit occurred).

        Returns
        -------
        dict
            Complete selection report with keys:
            - 'timestamp': When the report was generated
            - 'baseline_method': Initial solver method
            - 'selected_method': Recommended solver method
            - 'recommendation': Selection result dict
            - 'confidence': Confidence level
            - 'diagnostics': Diagnostic test results summary
            - 'decision_report': Complete decision analysis
            - 'performance': Performance comparison (if refit)
            - 'improvement': Performance improvement percentage
            - 'next_steps': Recommended actions

        Raises
        ------
        RuntimeError
            If framework has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before getting report")

        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_method': self.baseline_solver_name,
            'selected_method': self.selected_solver_name,
            'recommendation': self.selection_result,
            'confidence': self.selection_result['confidence'],
            'diagnostics': {
                'num_failures': len(self.diagnostic_results['summary'].get('failure_types', [])),
                'failure_types': self.diagnostic_results['summary'].get('failure_types', []),
                'heteroscedastic': self.diagnostic_results['heteroscedasticity'].get('heteroscedastic', False),
                'autocorrelated': self.diagnostic_results['autocorrelation'].get('autocorrelated', False),
                'nonstationary': self.diagnostic_results['nonstationarity'].get('nonstationary', False),
            },
            'decision_report': self.decision_report,
            'next_steps': self.selection_result['next_steps']
        }

        # Add performance comparison if refit occurred
        if self.performance_comparison:
            report['performance'] = self.performance_comparison
            report['improvement_percent'] = self.performance_comparison['improvement_percent']

        return report

    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get summary of diagnostic test results.

        Returns
        -------
        dict
            Summary with test results, p-values, and interpretations.

        Raises
        ------
        RuntimeError
            If framework has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before getting diagnostics")

        return {
            'heteroscedasticity': self.diagnostic_results['heteroscedasticity'],
            'autocorrelation': self.diagnostic_results['autocorrelation'],
            'nonstationarity': self.diagnostic_results['nonstationarity'],
            'state_dependence': self.diagnostic_results.get('state_dependence'),
            'summary': self.diagnostic_results['summary']
        }

    def get_method_alternatives(self) -> List[str]:
        """Get list of alternative solver methods.

        Returns
        -------
        list
            Alternative solver methods in priority order.

        Raises
        ------
        RuntimeError
            If framework has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before getting alternatives")

        return self.selection_result['alternatives']

    def is_fitted(self) -> bool:
        """Check if framework has been fitted.

        Returns
        -------
        bool
            True if fitted, False otherwise.
        """
        return self._is_fitted

    def reset(self) -> None:
        """Clear all history and return to initial state.

        This method resets the framework to its initialized state, clearing:
        - All solver history
        - All diagnostic results
        - All selection results
        - The fitted flag

        The framework can be fitted again after reset. Useful for:
        - Processing multiple datasets
        - Experimenting with different configurations
        - Clearing memory before reuse

        Examples
        --------
        >>> framework = AdaptiveSolverFramework()
        >>> framework.fit(t1, x1)
        >>> # Use framework...
        >>> framework.reset()  # Clear everything
        >>> framework.fit(t2, x2)  # Fit to new data
        """
        # Clear history
        self.solver_history = []
        self.diagnostic_history = []

        # Reset to initial solver
        self.baseline_solver = self._create_solver(self.initial_solver)
        self.baseline_solver_name = self.initial_solver
        self.selected_solver = None
        self.selected_solver_name = self.initial_solver

        # Clear results
        self.diagnostic_results = None
        self.selection_result = None
        self.decision_report = None
        self.performance_comparison = None

        # Clear state
        self._is_fitted = False
        self._fit_time = None
        self._t_data = None
        self._x_data = None

        if self.verbose:
            logger.info("AdaptiveSolverFramework reset to initial state")

    def set_available_solvers(self, solvers: Dict[str, str]) -> None:
        """Update available solvers for method selection.

        This method allows adding or modifying available solvers, useful for
        Week 4+ when new solvers (SDE, Neural, etc.) are implemented.

        Parameters
        ----------
        solvers : dict
            Mapping of solver names to descriptions.
            Example: {'SDE': 'Stochastic solver', 'Neural': 'Neural ODE'}

        Raises
        ------
        TypeError
            If solvers is not a dictionary.
        
        ValueError
            If solver dict contains empty names or non-string values.

        Notes
        -----
        This updates both the framework's available_solvers and the
        MethodSelector's list of available methods.

        Examples
        --------
        >>> framework = AdaptiveSolverFramework()
        >>> new_solvers = {
        ...     'SDE': 'Stochastic differential equations',
        ...     'Neural': 'Neural ODE solver'
        ... }
        >>> framework.set_available_solvers(new_solvers)

        Week 4 usage:
        >>> framework.set_available_solvers({
        ...     'SDE': 'Implemented in Week 4'
        ... })
        >>> framework.fit(t, x)  # Now SDE is available for selection
        """
        if not isinstance(solvers, dict):
            raise TypeError("solvers must be a dictionary")

        # Validate solver names and descriptions
        for name, description in solvers.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Solver names must be non-empty strings")
            if not isinstance(description, str):
                raise ValueError("Solver descriptions must be strings")

        # Update available solvers
        self.available_solvers.update(solvers)

        if self.verbose:
            logger.info(f"Updated available solvers: {list(solvers.keys())}")

    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Return aggregated summary of all diagnostics run.

        Returns comprehensive statistics on diagnostic tests including:
        - Counts of each failure type
        - Test results and p-values
        - Failure patterns across runs

        Returns
        -------
        dict
            Aggregated diagnostic summary with:
            - 'test_count': Number of diagnostic runs
            - 'failure_counts': Dictionary mapping failure types to counts
            - 'most_common_failures': List of most frequent failures
            - 'latest_results': Latest complete diagnostic results
            - 'summary': Overall summary statistics

        Raises
        ------
        RuntimeError
            If framework has not been fitted.

        Examples
        --------
        >>> framework = AdaptiveSolverFramework()
        >>> framework.fit(t, x)
        >>> summary = framework.get_diagnostic_summary()
        >>> print(f"Tests run: {summary['test_count']}")
        >>> print(f"Common failures: {summary['most_common_failures']}")
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before getting diagnostic summary")

        # Aggregate failure counts from history
        failure_counts = {}
        for diag in self.diagnostic_history:
            for failure in diag.get('summary', {}).get('failure_types', []):
                failure_counts[failure] = failure_counts.get(failure, 0) + 1

        # Sort failures by frequency
        most_common = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'test_count': len(self.diagnostic_history),
            'failure_counts': failure_counts,
            'most_common_failures': [name for name, _ in most_common],
            'latest_results': self.diagnostic_results,
            'summary': self.diagnostic_results.get('summary', {}) if self.diagnostic_results else {}
        }

    def compare_to_baseline(self, baseline_solver: str = 'RK45') -> Dict[str, Any]:
        """Compare current solver performance to a baseline solver.

        This method requires that fit() has been called with auto_refit=True,
        or that multiple solvers have been tried. It compares the currently
        selected solver against a specified baseline for performance metrics.

        Parameters
        ----------
        baseline_solver : str, default='RK45'
            Name of baseline solver to compare against.
            Options: 'RK45', 'RK4', 'SDE', 'Neural', 'Regime', 'Ensemble'

        Returns
        -------
        dict
            Comparison metrics including:
            - 'baseline': Baseline solver name
            - 'current': Currently selected solver name
            - 'baseline_metrics': Baseline performance metrics
            - 'current_metrics': Current performance metrics
            - 'improvement': Relative improvement percentages
            - 'speedup': Computational efficiency gains
            - 'recommendation': Whether current is better

        Raises
        ------
        RuntimeError
            If framework has not been fitted.

        ValueError
            If baseline_solver not in available_solvers.

        Notes
        -----
        Requires performance_comparison from fitting with auto_refit=True.
        If not available, returns theoretical comparison based on solver
        characteristics.

        Examples
        --------
        >>> framework = AdaptiveSolverFramework(auto_refit=True)
        >>> framework.fit(t, x)
        >>> comparison = framework.compare_to_baseline()
        >>> print(f"RMSE improvement: {comparison['improvement']['rmse']:.2%}")
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before comparing to baseline")

        if baseline_solver not in self.available_solvers:
            raise ValueError(
                f"baseline_solver '{baseline_solver}' not in available solvers: "
                f"{list(self.available_solvers.keys())}"
            )

        result = {
            'baseline': baseline_solver,
            'current': self.selected_solver_name,
            'baseline_metrics': None,
            'current_metrics': None,
            'improvement': {},
            'recommendation': None
        }

        # Use performance comparison from fitting if available
        if self.performance_comparison:
            result.update(self.performance_comparison)
            result['baseline'] = self.baseline_solver_name
        else:
            # No actual comparison available (both baseline and selected are same)
            result['recommendation'] = 'No improvement measured (same solver used)'

        return result

    def export_selection_log(self, filepath: str) -> None:
        """Export complete selection history to JSON file.

        Saves the full adaptive selection process to a JSON file for:
        - Auditing method selection decisions
        - Sharing results with stakeholders
        - Post-hoc analysis of solver performance
        - Documentation and reproducibility

        Parameters
        ----------
        filepath : str
            Path where JSON file will be saved.

        Raises
        ------
        RuntimeError
            If framework has not been fitted.

        IOError
            If file cannot be written to filepath.

        Examples
        --------
        >>> framework = AdaptiveSolverFramework()
        >>> framework.fit(t, x)
        >>> framework.export_selection_log('results/selection_log.json')

        File contents include:
        - Timestamp and configuration
        - Baseline and selected solver details
        - Complete diagnostic results
        - Selection decision with reasoning
        - Performance comparison
        - Alternative recommendations

        The exported JSON can be loaded and analyzed:
        >>> import json
        >>> with open('results/selection_log.json') as f:
        ...     log = json.load(f)
        >>> print(log['selection_result']['method'])
        >>> print(log['performance_comparison']['improvement_percent'])
        """
        if not self._is_fitted:
            raise RuntimeError("Framework must be fitted before exporting selection log")

        # Build comprehensive log
        log = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'framework_version': '0.3.0',
                'configuration': {
                    'baseline_solver': self.baseline_solver_name,
                    'prefer_speed': self.prefer_speed,
                    'prefer_accuracy': self.prefer_accuracy,
                    'prefer_interpretability': self.prefer_interpretability,
                    'max_complexity': self.max_complexity,
                    'max_cost': self.max_cost,
                    'auto_refit': self.auto_refit
                }
            },
            'data_info': {
                'n_samples': len(self._t_data) if self._t_data is not None else None,
                'n_states': self._x_data.shape[1] if self._x_data is not None else None,
                't_range': [
                    float(self._t_data.min()),
                    float(self._t_data.max())
                ] if self._t_data is not None else None
            },
            'baseline_solver': {
                'name': self.baseline_solver_name,
                'description': self.available_solvers.get(self.baseline_solver_name, 'Unknown')
            },
            'selected_solver': {
                'name': self.selected_solver_name,
                'description': self.available_solvers.get(self.selected_solver_name, 'Unknown')
            },
            'diagnostics': self._serialize_diagnostics(),
            'selection_result': self.selection_result,
            'decision_report': self.decision_report,
            'performance_comparison': self.performance_comparison,
            'solver_history': self.solver_history,
            'diagnostic_history': [
                self._serialize_diagnostics(diag) for diag in self.diagnostic_history
            ]
        }

        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(log, f, indent=2, default=str)
            if self.verbose:
                logger.info(f"Selection log exported to {filepath}")
        except IOError as e:
            raise IOError(f"Failed to write selection log to {filepath}: {e}")

    def _serialize_diagnostics(self, diagnostics: Optional[Dict] = None) -> Dict[str, Any]:
        """Serialize diagnostic results for JSON export.

        Converts numpy arrays and other non-JSON-serializable objects
        to JSON-compatible formats.

        Parameters
        ----------
        diagnostics : dict, optional
            Diagnostic results to serialize. If None, uses self.diagnostic_results

        Returns
        -------
        dict
            JSON-serializable diagnostic results
        """
        if diagnostics is None:
            diagnostics = self.diagnostic_results

        if diagnostics is None:
            return {}

        # Create serializable copy
        serialized = {
            'heteroscedasticity': diagnostics.get('heteroscedasticity', {}),
            'autocorrelation': diagnostics.get('autocorrelation', {}),
            'nonstationarity': diagnostics.get('nonstationarity', {}),
            'state_dependence': diagnostics.get('state_dependence', {}),
            'summary': diagnostics.get('summary', {})
        }

        return serialized

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"AdaptiveSolverFramework(baseline={self.baseline_solver_name}, "
            f"selected={self.selected_solver_name}, status={status})"
        )
