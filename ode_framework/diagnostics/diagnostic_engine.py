"""Diagnostic engine for ODE solver validation.

This module provides the main diagnostic engine that orchestrates multiple
statistical tests and generates comprehensive diagnostic reports. It helps
users understand model quality, identify potential issues, and debug
integration problems.

Features:
    - Run all statistical tests automatically
    - Generate comprehensive diagnostic reports
    - Suggest improvements based on test results
    - Identify common failure modes
    - Validate convergence of solver
    - Check for numerical stability issues

Example:
    >>> from ode_framework.diagnostics.diagnostic_engine import DiagnosticEngine
    >>> from ode_framework.solvers.classical import RK45Solver
    >>> engine = DiagnosticEngine(solver)
    >>> report = engine.run_diagnostics(t_data, x_data, t_eval)
    >>> print(report.summary())
"""

from typing import Dict, Optional, Any, List
import numpy as np
import logging

from .statistical_tests import (
    breusch_pagan_test,
    ljung_box_test,
    augmented_dickey_fuller_test,
    state_dependence_test,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DiagnosticReport:
    """Container for diagnostic test results.
    
    Attributes
    ----------
    tests_results : dict
        Dictionary of test names to (statistic, p_value) tuples
    metrics : dict
        Dictionary of diagnostic metrics
    warnings : list
        List of warning messages
    suggestions : list
        List of suggestions for improvement
    """
    
    def __init__(self) -> None:
        """Initialize empty diagnostic report."""
        self.tests_results: Dict[str, tuple] = {}
        self.metrics: Dict[str, float] = {}
        self.warnings: list = []
        self.suggestions: list = []
    
    def summary(self) -> str:
        """Generate human-readable summary of diagnostics.
        
        Returns
        -------
        str
            Formatted summary of all diagnostic results
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DIAGNOSTIC REPORT SUMMARY")
        lines.append("=" * 70)
        
        if self.tests_results:
            lines.append("\n📊 Statistical Tests:")
            lines.append("-" * 70)
            for test_name, result in self.tests_results.items():
                if isinstance(result, dict):
                    p_val = result.get('p_value', 'N/A')
                    flag = result.get(list(result.keys())[0], False) if len(result) > 2 else False
                    status = "⚠️ FAILED" if flag else "✓ PASSED"
                    lines.append(f"  {test_name:30s} {status:12s} (p={p_val:.4f})")
                else:
                    lines.append(f"  {test_name:30s} {str(result)}")
        
        if self.warnings:
            lines.append("\n⚠️  Warnings:")
            lines.append("-" * 70)
            for warning in self.warnings:
                lines.append(f"  • {warning}")
        
        if self.suggestions:
            lines.append("\n💡 Suggestions:")
            lines.append("-" * 70)
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")
        else:
            lines.append("\n✅ No issues detected. Model fit appears adequate.")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format.
        
        Returns
        -------
        dict
            Dictionary representation of the diagnostic report
        """
        return {
            'tests_results': self.tests_results,
            'metrics': self.metrics,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
        }


class DiagnosticEngine:
    """Main engine for running ODE solver diagnostics.
    
    Automatically runs statistical tests to validate solver fit quality,
    identify model issues, and provide actionable suggestions for improvement.
    
    Attributes
    ----------
    alpha : float
        Significance level for statistical tests (default: 0.05)
    results : dict
        Storage for diagnostic results
    
    Parameters
    ----------
    verbose : bool, optional
        If True, print progress messages. Default is False.
    """
    
    def __init__(self, verbose: bool = False) -> None:
        """Initialize diagnostic engine.
        
        Parameters
        ----------
        verbose : bool, optional
            Print progress messages
        """
        self.verbose = verbose
        self.alpha = 0.05  # Significance level
        self.results: Dict[str, Any] = {}
        if self.verbose:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
    
    def run_diagnostics(
        self,
        residuals: np.ndarray,
        time: np.ndarray,
        state_vars: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run all diagnostic tests on residuals.
        
        Executes all four statistical tests and aggregates results into a
        structured dictionary. Handles missing state_vars gracefully.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from ODE solver fit. Shape: (n_samples,) or (n_samples, 1)
        time : np.ndarray
            Time points corresponding to residuals. Shape: (n_samples,)
        state_vars : np.ndarray, optional
            State variables at each time point. Shape: (n_samples,) or (n_samples, n_states)
            If None, state_dependence_test is skipped.
        
        Returns
        -------
        dict
            Aggregated diagnostic results with keys:
            - 'heteroscedasticity': Breusch-Pagan test result (dict)
            - 'autocorrelation': Ljung-Box test result (dict)
            - 'nonstationarity': ADF test result (dict)
            - 'state_dependence': State-dependence test result (dict or None)
            - 'summary': Summary of detected issues (dict)
        
        Raises
        ------
        ValueError
            If residuals and time have mismatched lengths
        """
        # Input validation
        residuals = np.asarray(residuals).flatten()
        time = np.asarray(time).flatten()
        
        if len(residuals) != len(time):
            raise ValueError(
                f"residuals and time length mismatch: {len(residuals)} vs {len(time)}"
            )
        
        if len(residuals) < 3:
            raise ValueError(f"Need at least 3 observations, got {len(residuals)}")
        
        if self.verbose:
            logger.info(f"Running diagnostics on {len(residuals)} observations")
        
        # Run Breusch-Pagan test
        if self.verbose:
            logger.info("Running Breusch-Pagan test for heteroscedasticity...")
        try:
            bp_result = breusch_pagan_test(residuals, time.reshape(-1, 1))
            self.results['heteroscedasticity'] = bp_result
        except Exception as e:
            logger.error(f"Breusch-Pagan test failed: {str(e)}")
            self.results['heteroscedasticity'] = {
                'error': str(e),
                'heteroscedastic': None,
                'test_name': 'Breusch-Pagan',
            }
        
        # Run Ljung-Box test
        if self.verbose:
            logger.info("Running Ljung-Box test for autocorrelation...")
        try:
            lb_result = ljung_box_test(residuals, lags=10)
            self.results['autocorrelation'] = lb_result
        except Exception as e:
            logger.error(f"Ljung-Box test failed: {str(e)}")
            self.results['autocorrelation'] = {
                'error': str(e),
                'autocorrelated': None,
                'test_name': 'Ljung-Box',
            }
        
        # Run ADF test
        if self.verbose:
            logger.info("Running Augmented Dickey-Fuller test for stationarity...")
        try:
            adf_result = augmented_dickey_fuller_test(residuals)
            self.results['nonstationarity'] = adf_result
        except Exception as e:
            logger.error(f"ADF test failed: {str(e)}")
            self.results['nonstationarity'] = {
                'error': str(e),
                'nonstationary': None,
                'test_name': 'Augmented Dickey-Fuller',
            }
        
        # Run state-dependence test if state_vars provided
        if state_vars is not None:
            if self.verbose:
                logger.info("Running state-dependence test...")
            try:
                sd_result = state_dependence_test(residuals, state_vars)
                self.results['state_dependence'] = sd_result
            except Exception as e:
                logger.error(f"State-dependence test failed: {str(e)}")
                self.results['state_dependence'] = {
                    'error': str(e),
                    'state_dependent': None,
                    'test_name': 'State-Dependence',
                }
        else:
            self.results['state_dependence'] = None
        
        # Generate summary
        if self.verbose:
            logger.info("Generating summary...")
        self.results['summary'] = self._generate_summary()
        
        return self.results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Create human-readable summary of diagnostic results.
        
        Analyzes test results and generates actionable recommendations based
        on which tests failed (p_value < 0.05).
        
        Returns
        -------
        dict
            Summary dictionary with keys:
            - 'failure_detected': bool, True if any test shows failure
            - 'failure_types': list of str, names of detected issues
            - 'recommended_method': str, suggestion for model improvement
            - 'confidence': str, 'high', 'medium', or 'low'
        
        Notes
        -----
        Decision logic:
        - If all tests pass: "Classical solver adequate"
        - If heteroscedastic only: "Consider SDE with learned diffusion"
        - If autocorrelated only: "Consider Neural ODE"
        - If both heteroscedastic and autocorrelated: "Consider Neural ODE with stochastic"
        - If nonstationary: "Consider regime-switching model"
        - Multiple failures: "Consider ensemble approach"
        """
        failure_types: List[str] = []
        
        # Check each test result
        if 'heteroscedasticity' in self.results and self.results['heteroscedasticity']:
            result = self.results['heteroscedasticity']
            if 'error' not in result and result.get('heteroscedastic', False):
                failure_types.append('heteroscedastic')
        
        if 'autocorrelation' in self.results and self.results['autocorrelation']:
            result = self.results['autocorrelation']
            if 'error' not in result and result.get('autocorrelated', False):
                failure_types.append('autocorrelated')
        
        if 'nonstationarity' in self.results and self.results['nonstationarity']:
            result = self.results['nonstationarity']
            if 'error' not in result and result.get('nonstationary', False):
                failure_types.append('nonstationary')
        
        if self.results.get('state_dependence') and self.results['state_dependence']:
            result = self.results['state_dependence']
            if 'error' not in result and result.get('state_dependent', False):
                failure_types.append('state_dependent')
        
        # Determine failure status and recommendation
        failure_detected = len(failure_types) > 0
        
        if not failure_detected:
            recommended_method = "Classical solver appears adequate for this system"
            confidence = "high"
        elif failure_types == ['heteroscedastic']:
            recommended_method = "Consider Stochastic Differential Equation (SDE) with learned diffusion"
            confidence = "high"
        elif failure_types == ['autocorrelated']:
            recommended_method = "Consider Neural ODE for improved temporal dynamics"
            confidence = "medium"
        elif set(failure_types) == {'heteroscedastic', 'autocorrelated'}:
            recommended_method = "Consider Neural ODE with stochastic component"
            confidence = "medium"
        elif 'nonstationary' in failure_types:
            recommended_method = "Consider regime-switching model or time-varying parameters"
            confidence = "medium"
        elif len(failure_types) > 2:
            recommended_method = "Consider ensemble approach combining multiple model types"
            confidence = "low"
        else:
            recommended_method = "Review solver parameters and data quality"
            confidence = "medium"
        
        return {
            'failure_detected': failure_detected,
            'failure_types': failure_types,
            'recommended_method': recommended_method,
            'confidence': confidence,
            'num_failures': len(failure_types),
        }
    
    def generate_report(self) -> str:
        """Generate formatted text report of diagnostic results.
        
        Creates a comprehensive, human-readable report including:
        - All test results with p-values
        - Summary of detected failures
        - Recommended improvements
        - Confidence assessment
        
        Returns
        -------
        str
            Multi-line formatted report
        
        Raises
        ------
        RuntimeError
            If no diagnostics have been run yet
        """
        if not self.results:
            raise RuntimeError("No diagnostics run yet. Call run_diagnostics() first.")
        
        lines = []
        lines.append("=" * 75)
        lines.append("ODE SOLVER DIAGNOSTIC REPORT")
        lines.append("=" * 75)
        
        # Test Results Table
        lines.append("\n📊 STATISTICAL TEST RESULTS")
        lines.append("-" * 75)
        lines.append(f"{'Test':<30} {'Result':<15} {'P-Value':<15} {'Status':<15}")
        lines.append("-" * 75)
        
        # Breusch-Pagan
        if 'heteroscedasticity' in self.results:
            result = self.results['heteroscedasticity']
            if 'error' not in result:
                status = "⚠️ FAIL" if result.get('heteroscedastic') else "✓ PASS"
                p_val = f"{result.get('p_value', 0):.4f}"
            else:
                status = "❌ ERROR"
                p_val = "N/A"
            lines.append(f"{'Heteroscedasticity':<30} {'BP Test':<15} {p_val:<15} {status:<15}")
        
        # Ljung-Box
        if 'autocorrelation' in self.results:
            result = self.results['autocorrelation']
            if 'error' not in result:
                status = "⚠️ FAIL" if result.get('autocorrelated') else "✓ PASS"
                p_val = f"{result.get('p_value', 0):.4f}"
            else:
                status = "❌ ERROR"
                p_val = "N/A"
            lines.append(f"{'Autocorrelation':<30} {'LB Test':<15} {p_val:<15} {status:<15}")
        
        # ADF
        if 'nonstationarity' in self.results:
            result = self.results['nonstationarity']
            if 'error' not in result:
                status = "⚠️ FAIL" if result.get('nonstationary') else "✓ PASS"
                p_val = f"{result.get('p_value', 0):.4f}"
            else:
                status = "❌ ERROR"
                p_val = "N/A"
            lines.append(f"{'Stationarity':<30} {'ADF Test':<15} {p_val:<15} {status:<15}")
        
        # State-Dependence
        if self.results.get('state_dependence'):
            result = self.results['state_dependence']
            if 'error' not in result:
                status = "⚠️ FAIL" if result.get('state_dependent') else "✓ PASS"
                p_val = f"{result.get('p_value', 0):.4f}"
            else:
                status = "❌ ERROR"
                p_val = "N/A"
            lines.append(f"{'State Dependence':<30} {'SD Test':<15} {p_val:<15} {status:<15}")
        
        # Summary Section
        if 'summary' in self.results:
            summary = self.results['summary']
            lines.append("\n" + "=" * 75)
            lines.append("📋 SUMMARY")
            lines.append("=" * 75)
            
            # Detected issues line
            issues_str = ", ".join(summary['failure_types']) if summary['failure_types'] else "None"
            lines.append(f"\nDetected Issues: {issues_str}")
            
            lines.append(f"Failure Detected: {'Yes' if summary['failure_detected'] else 'No'}")
            lines.append(f"Number of Failures: {summary['num_failures']}")
            lines.append(f"Confidence Level: {summary['confidence'].upper()}")
            
            lines.append("\n" + "-" * 75)
            lines.append("💡 RECOMMENDATION")
            lines.append("-" * 75)
            lines.append(summary['recommended_method'])
            
            # Add confidence-based suggestions
            if summary['confidence'] == 'low':
                lines.append("\n⚠️  Note: Low confidence recommendation. Consider:")
                lines.append("  • Collecting more data")
                lines.append("  • Validating data quality")
                lines.append("  • Consulting domain experts")
        
        lines.append("\n" + "=" * 75)
        return "\n".join(lines)
    
    def identify_issues(self) -> List[str]:
        """Identify potential issues based on test results.
        
        Extracts specific issues from diagnostic results and provides
        detailed explanations.
        
        Returns
        -------
        list of str
            List of identified issues with descriptions
        
        Raises
        ------
        RuntimeError
            If no diagnostics have been run yet
        """
        if not self.results:
            raise RuntimeError("No diagnostics run yet. Call run_diagnostics() first.")
        
        issues = []
        
        if 'heteroscedasticity' in self.results:
            result = self.results['heteroscedasticity']
            if 'error' not in result and result.get('heteroscedastic'):
                issues.append(
                    "Heteroscedasticity: Residual variance depends on system state. "
                    "Consider state-dependent error models."
                )
        
        if 'autocorrelation' in self.results:
            result = self.results['autocorrelation']
            if 'error' not in result and result.get('autocorrelated'):
                sig_lags = result.get('significant_lags', [])
                issues.append(
                    f"Autocorrelation: Residuals at lags {sig_lags} are correlated. "
                    "ODE may not capture all temporal dependencies."
                )
        
        if 'nonstationarity' in self.results:
            result = self.results['nonstationarity']
            if 'error' not in result and result.get('nonstationary'):
                issues.append(
                    "Nonstationarity: Residual properties change over time. "
                    "System may have time-varying parameters or external forcing."
                )
        
        if self.results.get('state_dependence'):
            result = self.results['state_dependence']
            if 'error' not in result and result.get('state_dependent'):
                r2 = result.get('r_squared', 0)
                issues.append(
                    f"State Dependence: Error magnitude correlates with state (R²={r2:.3f}). "
                    "Model may need state-dependent refinement."
                )
        
        return issues
    
    def suggest_improvements(self) -> List[str]:
        """Generate suggestions for improving model fit quality.
        
        Based on identified issues and test results, provides specific,
        actionable suggestions for improvement.
        
        Returns
        -------
        list of str
            List of actionable suggestions
        
        Raises
        ------
        RuntimeError
            If no diagnostics have been run yet
        """
        if not self.results:
            raise RuntimeError("No diagnostics run yet. Call run_diagnostics() first.")
        
        suggestions = []
        summary = self.results.get('summary', {})
        failure_types = summary.get('failure_types', [])
        
        if not failure_types:
            suggestions.append("✓ Model fit appears adequate. No major improvements needed.")
            return suggestions
        
        # Heteroscedasticity suggestions
        if 'heteroscedastic' in failure_types:
            suggestions.append(
                "1. Implement adaptive error model: Use SDE with state-dependent diffusion"
            )
            suggestions.append("   - Fit diffusion coefficient σ(x) to data")
            suggestions.append("   - Use weighted residuals based on state values")
        
        # Autocorrelation suggestions
        if 'autocorrelated' in failure_types:
            suggestions.append(
                "2. Improve temporal dynamics: Consider Neural ODE architecture"
            )
            suggestions.append("   - Add hidden layers to capture unmodeled dynamics")
            suggestions.append("   - Increase model complexity using learned latent dynamics")
        
        # Nonstationarity suggestions
        if 'nonstationary' in failure_types:
            suggestions.append(
                "3. Handle time-varying behavior: Implement regime-switching model"
            )
            suggestions.append("   - Identify regime transitions in the data")
            suggestions.append("   - Fit separate ODE parameters for each regime")
        
        # State dependence suggestions
        if 'state_dependent' in failure_types:
            suggestions.append(
                "4. Refine state relationships: Learn state-dependent parameters"
            )
            suggestions.append("   - Parametrize ODE coefficients as functions of state")
            suggestions.append("   - Use neural networks to learn parameter manifolds")
        
        # General suggestions
        if len(failure_types) > 1:
            suggestions.append("\n5. Consider ensemble approach:")
            suggestions.append("   - Combine multiple model types (ODE + SDE + Neural ODE)")
            suggestions.append("   - Use mixture-of-experts for adaptive selection")
        
        return suggestions
