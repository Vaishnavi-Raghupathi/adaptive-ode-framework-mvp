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
