"""Decision logic for automatic solver selection based on diagnostic results.

This module implements rule-based decision trees that map diagnostic
failure patterns to appropriate numerical methods. The core logic
systematizes expert intuition about when to use classical solvers
versus more sophisticated approaches (SDE, Neural ODE).

The MethodSelector uses diagnostic test results from the Week 2 module
to automatically recommend the most appropriate solver for a given ODE
system. Decisions are based on:

1. Number and type of diagnostic failures
2. Severity of failures (p-value thresholds)
3. Combination of failures (e.g., heteroscedasticity + autocorrelation)
4. User-specified constraints (speed, accuracy, interpretability)

Key Classes
-----------
MethodSelector
    Main decision engine that maps diagnostic patterns to solver methods.

Decision Criteria
-----------------
The selection logic follows this hierarchy:

1. No failures → Classical solver (RK4, RK45) adequate
2. Heteroscedastic only → SDE with learned diffusion
3. Autocorrelated only → Neural ODE or regime-switching
4. Nonstationary → Time-varying parameters or regime-switching
5. State-dependent → Local error modeling or Neural ODE
6. Multiple failures → Most sophisticated method available

Design Principles
-----------------
- Systematic: Maps diagnostic patterns to methods using rule tables
- Transparent: Each recommendation explains the reasoning
- Flexible: Allows user constraints and custom rules
- Scalable: Easy to add new methods and rules

Example
-------
>>> from ode_framework.decision import MethodSelector
>>> from ode_framework.diagnostics import DiagnosticEngine
>>> import numpy as np

>>> # Get diagnostic results
>>> engine = DiagnosticEngine()
>>> results = engine.run_diagnostics(residuals, t_data)

>>> # Get method recommendation
>>> selector = MethodSelector()
>>> recommendation = selector.select_method(results)
>>> print(recommendation['method'])  # "Classical", "SDE", "Neural", etc.
>>> print(recommendation['reasoning'])
"""

from typing import Dict, List, Any, Optional
import numpy as np


class MethodSelector:
    """Automatic solver selection based on diagnostic test results.

    This class maps diagnostic failure patterns to appropriate ODE solver
    methods. It implements rule-based decision logic that systematizes
    expert knowledge about solver selection.

    Attributes
    ----------
    method_options : dict
        Available solver methods and their characteristics:
        - 'Classical': Fast, low computational cost, limited to simple dynamics
        - 'SDE': Handles stochastic noise, moderate computational cost
        - 'Neural': Universal approximator, high computational cost
        - 'Regime': Handles regime-switching, moderate cost
        - 'Ensemble': Combines multiple methods, computational cost varies
    
    thresholds : dict
        P-value thresholds for determining test failure:
        - Default alpha: 0.05 (5% significance level)
        - Can be customized per instantiation

    Rules
    -----
    Decision rules are applied in priority order:
    1. Check number of failures
    2. Check failure types and combinations
    3. Check p-value magnitudes
    4. Apply user constraints
    5. Return recommendation

    Raises
    ------
    ValueError
        If diagnostic results have invalid structure or values.

    Examples
    --------
    Create a selector and recommend a method:

    >>> selector = MethodSelector()
    >>> results = {
    ...     'heteroscedasticity': {'heteroscedastic': True, 'p_value': 0.01},
    ...     'autocorrelation': {'autocorrelated': False, 'p_value': 0.50},
    ...     'nonstationarity': {'nonstationary': False, 'p_value': 0.30},
    ...     'state_dependence': {'state_dependent': False, 'p_value': 0.80},
    ...     'summary': {'failure_types': ['heteroscedastic']}
    ... }
    >>> recommendation = selector.select_method(results)
    >>> print(recommendation['method'])  # "SDE"
    >>> print(recommendation['confidence'])  # "high"

    With custom constraints:

    >>> selector = MethodSelector(
    ...     prefer_speed=True,
    ...     max_complexity='moderate'
    ... )
    >>> recommendation = selector.select_method(results, prefer_speed=True)
    """

    # Method characteristics
    METHOD_CHARACTERISTICS = {
        'Classical': {
            'speed': 'very_fast',
            'complexity': 'low',
            'accuracy': 'good',
            'interpretability': 'high',
            'cost': 'low',
            'suited_for': ['no_failures', 'clean_data']
        },
        'SDE': {
            'speed': 'fast',
            'complexity': 'moderate',
            'accuracy': 'very_good',
            'interpretability': 'moderate',
            'cost': 'moderate',
            'suited_for': ['heteroscedastic', 'stochastic_noise']
        },
        'Neural': {
            'speed': 'slow',
            'complexity': 'high',
            'accuracy': 'excellent',
            'interpretability': 'low',
            'cost': 'high',
            'suited_for': ['complex_dynamics', 'multiple_failures']
        },
        'Regime': {
            'speed': 'moderate',
            'complexity': 'moderate',
            'accuracy': 'very_good',
            'interpretability': 'moderate',
            'cost': 'moderate',
            'suited_for': ['nonstationary', 'regime_switching']
        },
        'Ensemble': {
            'speed': 'slow',
            'complexity': 'high',
            'accuracy': 'excellent',
            'interpretability': 'moderate',
            'cost': 'high',
            'suited_for': ['uncertain_dynamics', 'all_failures']
        }
    }

    def __init__(
        self,
        alpha: float = 0.05,
        prefer_speed: bool = False,
        prefer_accuracy: bool = False,
        prefer_interpretability: bool = False,
        max_complexity: Optional[str] = None,
        max_cost: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the MethodSelector.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for determining test failure.
            Tests with p_value < alpha are considered failures.

        prefer_speed : bool, default=False
            If True, prefer faster methods over slower ones.
            
        prefer_accuracy : bool, default=False
            If True, prefer more accurate methods.
            
        prefer_interpretability : bool, default=False
            If True, prefer more interpretable methods.
            
        max_complexity : {'low', 'moderate', 'high'}, optional
            Maximum allowed complexity. If specified, rules out
            methods exceeding this complexity level.
            
        max_cost : {'low', 'moderate', 'high'}, optional
            Maximum computational cost. If specified, rules out
            expensive methods.
            
        verbose : bool, default=False
            If True, print detailed decision reasoning.
        """
        self.alpha = alpha
        self.prefer_speed = prefer_speed
        self.prefer_accuracy = prefer_accuracy
        self.prefer_interpretability = prefer_interpretability
        self.max_complexity = max_complexity
        self.max_cost = max_cost
        self.verbose = verbose

    def select_method(
        self,
        diagnostic_results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Select the most appropriate solver method.

        Analyzes diagnostic results and returns a method recommendation
        with confidence level and reasoning.

        Parameters
        ----------
        diagnostic_results : dict
            Output from DiagnosticEngine.run_diagnostics(). Must contain:
            - 'heteroscedasticity': dict with 'heteroscedastic' bool
            - 'autocorrelation': dict with 'autocorrelated' bool
            - 'nonstationarity': dict with 'nonstationary' bool
            - 'state_dependence': dict with 'state_dependent' bool (optional)
            - 'summary': dict with 'failure_types' list

        **kwargs
            Temporary overrides for instance preferences.

        Returns
        -------
        dict
            Recommendation dictionary with keys:
            - 'method': str, recommended method name
            - 'confidence': str, 'high', 'medium', or 'low'
            - 'reasoning': str, explanation of recommendation
            - 'alternatives': list, other viable methods
            - 'next_steps': list, suggested actions
            - 'p_values': dict, relevant p-values from tests

        Raises
        ------
        ValueError
            If diagnostic_results has invalid structure.
        """
        # Validate input
        self._validate_diagnostic_results(diagnostic_results)

        # Extract failure information
        failures = self._extract_failures(diagnostic_results)

        # Apply constraints
        viable_methods = self._apply_constraints(failures)

        # Apply decision rules
        selected_method = self._apply_decision_rules(failures, viable_methods)

        # Determine confidence
        confidence = self._determine_confidence(failures, selected_method)

        # Generate reasoning
        reasoning = self._generate_reasoning(failures, selected_method)

        # Find alternatives
        alternatives = self._find_alternatives(failures, viable_methods, selected_method)

        # Generate next steps
        next_steps = self._generate_next_steps(failures, selected_method)

        return {
            'method': selected_method,
            'confidence': confidence,
            'reasoning': reasoning,
            'alternatives': alternatives,
            'next_steps': next_steps,
            'p_values': {
                'heteroscedasticity': diagnostic_results['heteroscedasticity'].get('p_value'),
                'autocorrelation': diagnostic_results['autocorrelation'].get('p_value'),
                'nonstationarity': diagnostic_results['nonstationarity'].get('p_value'),
                'state_dependence': (diagnostic_results.get('state_dependence') or {}).get('p_value')
            }
        }

    def _validate_diagnostic_results(self, results: Dict[str, Any]) -> None:
        """Validate diagnostic results structure.

        Parameters
        ----------
        results : dict
            Diagnostic results to validate.

        Raises
        ------
        ValueError
            If structure is invalid.
        """
        required_keys = ['heteroscedasticity', 'autocorrelation', 'nonstationarity', 'summary']
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")

        # Validate test result structure
        for test_name in ['heteroscedasticity', 'autocorrelation', 'nonstationarity']:
            test_result = results[test_name]
            if not isinstance(test_result, dict):
                raise ValueError(f"{test_name} should be a dict")
            if 'p_value' not in test_result:
                raise ValueError(f"{test_name} missing p_value")

        # Validate summary
        if 'failure_types' not in results['summary']:
            raise ValueError("summary missing failure_types")

    def _extract_failures(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Extract failure flags from diagnostic results.

        Parameters
        ----------
        results : dict
            Diagnostic results.

        Returns
        -------
        dict
            Failure status for each test:
            - 'heteroscedastic': bool
            - 'autocorrelated': bool
            - 'nonstationary': bool
            - 'state_dependent': bool
            - 'num_failures': int
        """
        failures = {
            'heteroscedastic': results['heteroscedasticity'].get('heteroscedastic', False),
            'autocorrelated': results['autocorrelation'].get('autocorrelated', False),
            'nonstationary': results['nonstationarity'].get('nonstationary', False),
            'state_dependent': (results.get('state_dependence') or {}).get('state_dependent', False),
        }

        failures['num_failures'] = sum([
            failures['heteroscedastic'],
            failures['autocorrelated'],
            failures['nonstationary'],
            failures['state_dependent']
        ])

        return failures

    def _apply_constraints(self, failures: Dict[str, Any]) -> List[str]:
        """Apply user constraints to filter available methods.

        Parameters
        ----------
        failures : dict
            Failure status from diagnostic results.

        Returns
        -------
        list
            Methods that satisfy all constraints.
        """
        viable = list(self.METHOD_CHARACTERISTICS.keys())

        # Apply complexity constraint
        if self.max_complexity:
            complexity_order = {'low': 0, 'moderate': 1, 'high': 2}
            max_level = complexity_order.get(self.max_complexity, 2)
            viable = [
                m for m in viable
                if complexity_order.get(self.METHOD_CHARACTERISTICS[m]['complexity'], 2) <= max_level
            ]

        # Apply cost constraint
        if self.max_cost:
            cost_order = {'low': 0, 'moderate': 1, 'high': 2}
            max_level = cost_order.get(self.max_cost, 2)
            viable = [
                m for m in viable
                if cost_order.get(self.METHOD_CHARACTERISTICS[m]['cost'], 2) <= max_level
            ]

        return viable

    def _apply_decision_rules(self, failures: Dict[str, Any], viable: List[str]) -> str:
        """Apply decision rules to select best method.

        Parameters
        ----------
        failures : dict
            Failure status from diagnostic results.
        viable : list
            Methods that satisfy constraints.

        Returns
        -------
        str
            Recommended method name.
        """
        # Rule 1: No failures → Classical
        if failures['num_failures'] == 0:
            return 'Classical' if 'Classical' in viable else viable[0]

        # Rule 2: Only heteroscedasticity → SDE
        if failures['heteroscedastic'] and not any([
            failures['autocorrelated'],
            failures['nonstationary'],
            failures['state_dependent']
        ]):
            return 'SDE' if 'SDE' in viable else 'Neural'

        # Rule 3: Only autocorrelation → Neural or Regime
        if failures['autocorrelated'] and not any([
            failures['heteroscedastic'],
            failures['nonstationary'],
            failures['state_dependent']
        ]):
            return 'Neural' if 'Neural' in viable else 'Regime'

        # Rule 4: Only nonstationarity → Regime
        if failures['nonstationary'] and not any([
            failures['heteroscedastic'],
            failures['autocorrelated'],
            failures['state_dependent']
        ]):
            return 'Regime' if 'Regime' in viable else 'Neural'

        # Rule 5: Multiple failures → Most sophisticated
        if failures['num_failures'] >= 2:
            if 'Ensemble' in viable:
                return 'Ensemble'
            elif 'Neural' in viable:
                return 'Neural'
            else:
                return viable[-1] if viable else 'Classical'

        # Default: Neural
        return 'Neural' if 'Neural' in viable else viable[-1] if viable else 'Classical'

    def _determine_confidence(self, failures: Dict[str, Any], method: str) -> str:
        """Determine confidence level of recommendation.

        Parameters
        ----------
        failures : dict
            Failure status.
        method : str
            Selected method.

        Returns
        -------
        str
            Confidence level: 'high', 'medium', or 'low'.
        """
        if failures['num_failures'] == 0:
            return 'high'
        elif failures['num_failures'] == 1:
            return 'high'
        elif failures['num_failures'] == 2:
            return 'medium'
        else:
            return 'low' if method in ['Neural', 'Ensemble'] else 'medium'

    def _generate_reasoning(self, failures: Dict[str, Any], method: str) -> str:
        """Generate explanation for the recommendation.

        Parameters
        ----------
        failures : dict
            Failure status.
        method : str
            Selected method.

        Returns
        -------
        str
            Reasoning explanation.
        """
        if failures['num_failures'] == 0:
            return "No diagnostic failures detected. Classical solver is appropriate."

        failure_list = [k for k, v in failures.items() if v and k != 'num_failures']
        failure_text = ', '.join(failure_list)

        reasons = {
            'Classical': f"No issues detected. Classical solver sufficient.",
            'SDE': f"Heteroscedasticity detected. SDE can model varying noise.",
            'Neural': f"Complex failure pattern ({failure_text}). Neural ODE can learn complex dynamics.",
            'Regime': f"Nonstationarity detected. Regime-switching handles parameter changes.",
            'Ensemble': f"Multiple failures ({failure_text}). Ensemble combines approaches.",
        }

        return reasons.get(method, f"Recommended {method} based on diagnostic results.")

    def _find_alternatives(self, failures: Dict[str, Any], viable: List[str], selected: str) -> List[str]:
        """Find alternative methods.

        Parameters
        ----------
        failures : dict
            Failure status.
        viable : list
            Viable methods.
        selected : str
            Selected method.

        Returns
        -------
        list
            Alternative method names (sorted by relevance).
        """
        alternatives = [m for m in viable if m != selected]
        
        # Sort alternatives by approximate relevance
        relevance_order = ['Classical', 'SDE', 'Regime', 'Neural', 'Ensemble']
        alternatives.sort(key=lambda x: relevance_order.index(x) if x in relevance_order else 999)
        
        return alternatives[:2]  # Return top 2 alternatives

    def _generate_next_steps(self, failures: Dict[str, Any], method: str) -> List[str]:
        """Generate suggested next steps.

        Parameters
        ----------
        failures : dict
            Failure status.
        method : str
            Selected method.

        Returns
        -------
        list
            Suggested actions.
        """
        steps = [f"Implement {method} solver"]

        if failures['heteroscedastic']:
            steps.append("Investigate variance patterns in residuals")

        if failures['autocorrelated']:
            steps.append("Check for missed system dynamics or missing variables")

        if failures['nonstationary']:
            steps.append("Analyze time-varying behavior of system parameters")

        if failures['state_dependent']:
            steps.append("Model residual dependence on system state")

        steps.append("Validate improved solver with diagnostic tests")

        return steps

    def compute_confidence(
        self,
        diagnostic_results: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Assess confidence in recommendation based on diagnostic evidence.

        Confidence is determined by evaluating:
        1. Number of failures (fewer failures = higher confidence)
        2. P-value magnitudes (stronger evidence = higher confidence)
        3. Consistency of failure pattern (clear patterns = higher confidence)
        4. Statistical power (sufficient data = higher confidence)

        Confidence Criteria
        -------------------
        HIGH CONFIDENCE:
        - Single clear failure with strong statistical evidence (p < 0.01)
        - No failures (clean data)
        - Consistent failure pattern matching selected method

        MEDIUM CONFIDENCE:
        - 2-3 failures with moderate evidence (0.01 < p < 0.05)
        - Single failure with moderate evidence (0.05 < p < 0.10)
        - Multiple failures with inconsistent patterns

        LOW CONFIDENCE:
        - 4+ failures (highly uncertain system)
        - Weak statistical evidence across multiple tests (p > 0.10)
        - Edge cases or conflicting failure patterns

        Parameters
        ----------
        diagnostic_results : dict
            Output from DiagnosticEngine.run_diagnostics().
        recommendation : str
            Selected method name.

        Returns
        -------
        str
            Confidence level: 'high', 'medium', or 'low'.

        Notes
        -----
        This is an extended version of _determine_confidence that considers
        p-value magnitudes and consistency for more nuanced assessment.
        """
        # Extract p-values
        hetero_p = diagnostic_results['heteroscedasticity'].get('p_value', 1.0)
        auto_p = diagnostic_results['autocorrelation'].get('p_value', 1.0)
        nonstat_p = diagnostic_results['nonstationarity'].get('p_value', 1.0)
        state_p = (diagnostic_results.get('state_dependence') or {}).get('p_value', 1.0)

        # Extract failure flags
        failures = self._extract_failures(diagnostic_results)
        num_failures = failures['num_failures']

        # No failures → HIGH confidence
        if num_failures == 0:
            return 'high'

        # Single failure → Check p-value strength
        if num_failures == 1:
            p_values = [hetero_p, auto_p, nonstat_p, state_p]
            min_p = min([p for p in p_values if p < 0.5])  # Get actual test p-value

            if min_p < 0.01:
                return 'high'  # Very strong evidence
            elif min_p < 0.05:
                return 'medium'  # Moderate evidence
            else:
                return 'low'  # Weak evidence

        # Two to three failures → Check consistency
        if 2 <= num_failures <= 3:
            # Check if failures are consistent with recommendation
            if recommendation in ['Neural', 'Ensemble']:
                return 'medium'  # Complex methods handle multiple failures
            
            # Check p-value magnitudes
            p_values = [hetero_p, auto_p, nonstat_p, state_p]
            strong_evidence = sum([p < 0.01 for p in p_values])
            
            if strong_evidence >= 2:
                return 'high'  # Multiple strong signals
            else:
                return 'medium'

        # Four or more failures → LOW confidence
        if num_failures >= 4:
            return 'low'

        return 'medium'  # Default fallback

    def get_alternative_solvers(self, primary_recommendation: str) -> List[str]:
        """Suggest alternative solvers in priority order.

        Provides a ranked list of alternative methods that could be used
        if the primary recommendation proves inadequate. Ordering is based on:
        1. Computational cost
        2. Accuracy vs. speed tradeoff
        3. Likelihood of addressing undiagnosed issues

        Alternative Strategy
        --------------------
        - If primary is simplistic (Classical) → suggest more sophisticated options
        - If primary is sophisticated → suggest less costly alternatives for validation
        - Always maintain at least one classical option as fallback

        Parameters
        ----------
        primary_recommendation : str
            The primary recommended method (e.g., 'SDE', 'Neural').

        Returns
        -------
        list
            Ordered list of alternative method names.
            Empty list if primary is the only option.

        Examples
        --------
        >>> selector = MethodSelector()
        >>> alternatives = selector.get_alternative_solvers('Neural')
        >>> print(alternatives)  # ['SDE', 'Regime', 'Classical']

        >>> alternatives = selector.get_alternative_solvers('Classical')
        >>> print(alternatives)  # ['SDE', 'Regime', 'Neural']
        """
        # Define fallback chains for each method
        fallback_chains = {
            'Classical': ['SDE', 'Regime', 'Neural', 'Ensemble'],
            'SDE': ['Regime', 'Neural', 'Classical', 'Ensemble'],
            'Regime': ['Neural', 'SDE', 'Ensemble', 'Classical'],
            'Neural': ['Regime', 'SDE', 'Ensemble', 'Classical'],
            'Ensemble': ['Neural', 'Regime', 'SDE', 'Classical']
        }

        # Get fallback chain, defaulting to all methods if not found
        if primary_recommendation in fallback_chains:
            alternatives = fallback_chains[primary_recommendation]
        else:
            # For unknown methods, return all others ordered by complexity
            all_methods = list(self.METHOD_CHARACTERISTICS.keys())
            alternatives = [m for m in all_methods if m != primary_recommendation]

        return alternatives

    def generate_decision_report(
        self,
        diagnostic_results: Dict[str, Any],
        selection_result: Dict[str, Any],
        include_confidence_breakdown: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive decision report for logging and debugging.

        Creates a detailed report documenting the decision process, including:
        - Complete diagnostic summary
        - Selected recommendation with confidence
        - Decision tree path (which rules were applied)
        - Confidence breakdown (reasons for confidence level)
        - Alternative options considered
        - P-value analysis

        This report is useful for:
        1. Debugging unexpected recommendations
        2. Auditing decision logic
        3. Communicating results to stakeholders
        4. Iterating on rule sets

        Parameters
        ----------
        diagnostic_results : dict
            Output from DiagnosticEngine.run_diagnostics().
        selection_result : dict
            Output from select_method() or recommend_method().
        include_confidence_breakdown : bool, default=True
            If True, include detailed confidence scoring breakdown.

        Returns
        -------
        dict
            Comprehensive decision report with keys:
            - 'timestamp': datetime of report generation
            - 'diagnostic_summary': Summary of test results
            - 'recommendation': Selected method
            - 'confidence': Confidence level with detailed breakdown
            - 'decision_tree_path': List of rules applied (in order)
            - 'p_value_analysis': Statistical evidence summary
            - 'alternatives': Alternative methods with rationales
            - 'next_steps': Recommended actions
            - 'reasoning': Full explanation

        Examples
        --------
        >>> from ode_framework.decision import recommend_method
        >>> from datetime import datetime
        >>> results = engine.run_diagnostics(residuals, t_data)
        >>> recommendation = recommend_method(results)
        >>> selector = MethodSelector()
        >>> report = selector.generate_decision_report(results, recommendation)
        >>> print(f"Recommendation: {report['recommendation']}")
        >>> print(f"Confidence: {report['confidence']['level']}")
        """
        from datetime import datetime

        # Extract failure information
        failures = self._extract_failures(diagnostic_results)

        # Determine which rules were applied (decision tree path)
        decision_path = self._determine_decision_path(failures)

        # Compute confidence with detailed breakdown
        if include_confidence_breakdown:
            confidence_breakdown = self._compute_confidence_breakdown(
                diagnostic_results,
                selection_result['method']
            )
        else:
            confidence_breakdown = {}

        # P-value analysis
        p_value_analysis = {
            'heteroscedasticity': {
                'p_value': diagnostic_results['heteroscedasticity'].get('p_value'),
                'detected': diagnostic_results['heteroscedasticity'].get('heteroscedastic', False),
                'interpretation': 'Variance changes over time' if diagnostic_results['heteroscedasticity'].get('heteroscedastic') else 'Constant variance'
            },
            'autocorrelation': {
                'p_value': diagnostic_results['autocorrelation'].get('p_value'),
                'detected': diagnostic_results['autocorrelation'].get('autocorrelated', False),
                'interpretation': 'Residuals are correlated' if diagnostic_results['autocorrelation'].get('autocorrelated') else 'Residuals are independent'
            },
            'nonstationarity': {
                'p_value': diagnostic_results['nonstationarity'].get('p_value'),
                'detected': diagnostic_results['nonstationarity'].get('nonstationary', False),
                'interpretation': 'Mean/variance changes over time' if diagnostic_results['nonstationarity'].get('nonstationary') else 'System is stationary'
            },
            'state_dependence': {
                'p_value': (diagnostic_results.get('state_dependence') or {}).get('p_value'),
                'detected': (diagnostic_results.get('state_dependence') or {}).get('state_dependent', False),
                'interpretation': 'Errors depend on state' if (diagnostic_results.get('state_dependence') or {}).get('state_dependent') else 'Errors are independent of state'
            }
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'diagnostic_summary': {
                'num_failures': failures['num_failures'],
                'failure_types': [k for k, v in failures.items() if v and k != 'num_failures'],
                'all_tests': diagnostic_results['summary'].get('failure_types', [])
            },
            'recommendation': selection_result['method'],
            'confidence': {
                'level': selection_result['confidence'],
                'breakdown': confidence_breakdown
            },
            'decision_tree_path': decision_path,
            'p_value_analysis': p_value_analysis,
            'alternatives': selection_result['alternatives'],
            'next_steps': selection_result['next_steps'],
            'reasoning': selection_result['reasoning'],
            'method_characteristics': self.METHOD_CHARACTERISTICS.get(
                selection_result['method'],
                {}
            )
        }

    def _determine_decision_path(self, failures: Dict[str, Any]) -> List[str]:
        """Determine which decision rules were applied.

        Parameters
        ----------
        failures : dict
            Failure status from _extract_failures().

        Returns
        -------
        list
            Ordered list of decision rules applied.
        """
        path = []

        if failures['num_failures'] == 0:
            path.append("Rule 1: No failures detected → Classical solver")
        elif failures['num_failures'] == 1:
            if failures['heteroscedastic']:
                path.append("Rule 2: Heteroscedasticity only → SDE recommended")
            elif failures['autocorrelated']:
                path.append("Rule 3: Autocorrelation only → Neural or Regime")
            elif failures['nonstationary']:
                path.append("Rule 4: Nonstationarity only → Regime recommended")
            elif failures['state_dependent']:
                path.append("Rule 5: State dependence only → Neural recommended")
        else:
            path.append(f"Rule 6: Multiple failures ({failures['num_failures']}) → Sophisticated method")

        return path

    def _compute_confidence_breakdown(
        self,
        diagnostic_results: Dict[str, Any],
        selected_method: str
    ) -> Dict[str, Any]:
        """Compute detailed confidence scoring breakdown.

        Parameters
        ----------
        diagnostic_results : dict
            Diagnostic results.
        selected_method : str
            Selected method.

        Returns
        -------
        dict
            Detailed breakdown of confidence factors.
        """
        # Extract p-values
        hetero_p = diagnostic_results['heteroscedasticity'].get('p_value', 1.0)
        auto_p = diagnostic_results['autocorrelation'].get('p_value', 1.0)
        nonstat_p = diagnostic_results['nonstationarity'].get('p_value', 1.0)
        state_p = (diagnostic_results.get('state_dependence') or {}).get('p_value', 1.0)

        failures = self._extract_failures(diagnostic_results)

        # Assess evidence strength
        strong_signals = sum([
            hetero_p < 0.01,
            auto_p < 0.01,
            nonstat_p < 0.01,
            state_p < 0.01
        ])

        moderate_signals = sum([
            0.01 <= hetero_p < 0.05,
            0.01 <= auto_p < 0.05,
            0.01 <= nonstat_p < 0.05,
            0.01 <= state_p < 0.05
        ])

        weak_signals = sum([
            hetero_p >= 0.05,
            auto_p >= 0.05,
            nonstat_p >= 0.05,
            state_p >= 0.05
        ])

        return {
            'num_failures': failures['num_failures'],
            'strong_signals': strong_signals,
            'moderate_signals': moderate_signals,
            'weak_signals': weak_signals,
            'evidence_quality': (
                'very_strong' if strong_signals >= 2 else
                'strong' if strong_signals == 1 else
                'moderate' if moderate_signals >= 2 else
                'weak'
            ),
            'method_fit': (
                'excellent' if failures['num_failures'] == 0 else
                'good' if failures['num_failures'] == 1 else
                'adequate' if failures['num_failures'] <= 3 else
                'uncertain'
            ),
            'recommendation': f"{selected_method} is {'well-suited' if strong_signals >= 1 else 'suitable'} for this pattern"
        }


# Convenience functions for quick selection

def recommend_method(diagnostic_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Quick function to get method recommendation.

    Parameters
    ----------
    diagnostic_results : dict
        Output from DiagnosticEngine.run_diagnostics().
    **kwargs
        Passed to MethodSelector.__init__().

    Returns
    -------
    dict
        Recommendation with method, confidence, reasoning, etc.

    Example
    -------
    >>> from ode_framework.decision import recommend_method
    >>> recommendation = recommend_method(results)
    >>> print(recommendation['method'])
    """
    selector = MethodSelector(**kwargs)
    return selector.select_method(diagnostic_results)
