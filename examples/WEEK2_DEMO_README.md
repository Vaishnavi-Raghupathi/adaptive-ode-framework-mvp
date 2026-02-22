# Week 2 Diagnostics Framework Demo

## Overview

The **Week 2 diagnostics demo** (`examples/week2_demo.py`) provides a complete, educational walkthrough of the ODE solver diagnostics framework. It demonstrates the full pipeline from data generation through diagnostic analysis and interpretation.

## What the Demo Does

### 1. **Generate Synthetic Test Data**
- Creates exponential decay ODE: `dx/dt = -0.5*x`
- Adds realistic 3% measurement noise
- 150 time points from 0 to 5 seconds
- Realistic scenario matching real-world applications

### 2. **Fit ODE Solver**
- Uses `RK45Solver` with SINDy system identification
- Learns the ODE model from noisy observations
- High-precision numerical integration (rtol=1e-6, atol=1e-9)
- Demonstrates Week 1 solver capabilities

### 3. **Compute Residuals**
- Computes prediction errors (actual - predicted)
- Calculates residual statistics (mean, std, min, max, RMSE)
- Prepares data for statistical analysis

### 4. **Run Comprehensive Diagnostics**
Executes all 4 statistical tests:
- **Breusch-Pagan Test**: Detects heteroscedasticity (varying variance)
- **Ljung-Box Test**: Detects autocorrelation (serial dependence)
- **ADF Test**: Detects non-stationarity (changing mean/variance)
- **State-Dependence Test**: Detects correlation with system state

### 5. **Generate Diagnostic Report**
- Formatted test result summary
- Failure type identification
- Recommended solver method
- Confidence assessment

### 6. **Create Visualizations**
Generates 2×2 diagnostic plot with:
- **Top-left**: Residuals over time (scatter plot)
- **Top-right**: Autocorrelation function (ACF)
- **Bottom-left**: Q-Q plot (normality check)
- **Bottom-right**: Variance trend analysis

### 7. **Provide Interpretation**
- Explains what each failure type means
- Suggests specific improvements
- Guides next steps in solver optimization

## Running the Demo

### Basic Execution

```bash
cd examples
python week2_demo.py
```

### Expected Output

**Console Output:**
- 8-step walkthrough with formatted headers
- Intermediate results and statistics
- Diagnostic test results with p-values
- Formatted summary table
- Interpretation and recommendations
- Total execution time: ~0.3-0.5 seconds

**Generated Files:**
- `examples/week2_diagnostics.png` - 2×2 diagnostic visualization (280+ KB)

### Sample Output Snippet

```
======================================================================
  WEEK 2: DIAGNOSTICS FRAMEWORK - FULL DEMONSTRATION
======================================================================

This demo showcases:
  • Fitting an ODE solver to noisy data
  • Computing and analyzing residuals
  • Running statistical diagnostic tests
  • Generating diagnostic visualizations
  • Interpreting diagnostic recommendations

[STEP 1] Generate synthetic exponential decay data with noise
----------------------------------------------------------------------
  • Time points: 150 (0 to 5 seconds)
  • ODE: dx/dt = -0.5*x, x(0) = 1.0
  • Noise level: 3% (realistic measurement uncertainty)
  • Data shape: (150, 1)
  ✓ Synthetic data generated successfully

[STEP 2] Fit RK45Solver to noisy data using SINDy system identification
----------------------------------------------------------------------
  • Solver: RK45Solver (adaptive Runge-Kutta 4/5)
  • Tolerance: rtol=1e-6, atol=1e-9
  • Training data points: 150
  ✓ Solver fitted successfully using SINDy

[STEP 3] Generate predictions and compute residuals
----------------------------------------------------------------------
  • Residuals statistics:
    - Mean: -0.054722 (ideally ~0)
    - Std Dev: 0.068002
    - RMSE: 0.087285

[STEP 4] Run comprehensive statistical diagnostic tests
----------------------------------------------------------------------
  Test Results:
  - Breusch-Pagan Test (Heteroscedasticity)     FAILED ✗
  - Ljung-Box Test (Autocorrelation)            FAILED ✗
  - ADF Test (Stationarity)                     FAILED ✗
  - State-Dependence Test                       FAILED ✗

[STEP 5] Generate diagnostic report and recommendations
----------------------------------------------------------------------
  ODE SOLVER DIAGNOSTIC REPORT
  
  📊 STATISTICAL TEST RESULTS
  
  Test                    Result      P-Value     Status
  Heteroscedasticity      BP Test     0.0000      ⚠️ FAIL
  Autocorrelation         LB Test     0.0000      ⚠️ FAIL
  Stationarity            ADF Test    0.9980      ⚠️ FAIL
  State Dependence        SD Test     0.0000      ⚠️ FAIL
  
  💡 RECOMMENDATION
  Consider regime-switching model or time-varying parameters

[STEP 6] Create diagnostic visualizations
----------------------------------------------------------------------
  ✓ Diagnostic plot saved successfully
  • Location: examples/week2_diagnostics.png
  • File size: 281.6 KB

[STEP 7] Formatted diagnostic summary table
----------------------------------------------------------------------
  DIAGNOSTIC TEST RESULTS
  ═════════════════════════════════════════════════════════
  Test Name                    Detected  P-Value    Status
  ─────────────────────────────────────────────────────────
  Breusch-Pagan (Heteroscedasticity)  YES  0.000000  FAILED ✗
  Ljung-Box (Autocorrelation)         YES  0.000000  FAILED ✗
  ADF Test (Stationarity)             YES  0.998023  FAILED ✗
  State-Dependence Test               YES  0.000000  FAILED ✗

[STEP 8] Interpretation and recommended next steps
----------------------------------------------------------------------
  ⚠ Issues Detected: heteroscedastic, autocorrelated, nonstationary, state_dependent
  
  This indicates that the current solver may not be capturing all aspects 
  of the system dynamics. Consider:

  • Heteroscedasticity detected (variance changes over time)
    → Increase solver tolerance (tighter integration)
    → Try different numerical method
    → Consider stochastic ODE (SDE) formulation

  • Autocorrelation detected (residuals are correlated)
    → Increase model complexity or add missing terms
    → Use machine learning-based identification (e.g., Neural ODE)
    → Check for unmodeled dynamics

  • [Additional recommendations...]

======================================================================
  DEMONSTRATION COMPLETE
======================================================================

Total execution time: 0.35 seconds
Output files:
  • Diagnostic plot: examples/week2_diagnostics.png
```

## Diagnostic Interpretation Guide

### When All Tests Pass ✓

**Meaning:** Residuals are well-behaved
- Errors are randomly distributed
- No systematic patterns detected
- Solver has captured the essential dynamics

**Recommendation:** 
- Model is adequate for this application
- Consider solver as baseline for improvements

### When Heteroscedasticity Fails

**Meaning:** Residual variance changes over time
- Errors are larger in some regions
- Could indicate model missing state-dependent terms
- May reflect increasing measurement uncertainty

**Solutions:**
- Increase solver precision/tolerance
- Add adaptive error control
- Use stochastic ODE formulation
- Model error as function of state

### When Autocorrelation Fails

**Meaning:** Residuals are not independent
- Errors today depend on errors yesterday
- Suggests missing model structure
- Indicates model incompleteness

**Solutions:**
- Increase model complexity
- Add missing terms to ODE
- Use Neural ODE for automatic discovery
- Check for unmodeled dynamics

### When Stationarity Fails

**Meaning:** Mean or variance changes over time
- System exhibits regime changes
- Non-stationary behavior detected
- May indicate time-varying parameters

**Solutions:**
- Use regime-switching models
- Add time-varying parameter terms
- Split data into stationary regimes
- Consider switching ODE formulation

### When State-Dependence Fails

**Meaning:** Errors correlate with state value
- Prediction accuracy depends on system state
- High values have larger errors, or vice versa
- Suggests state-dependent error model needed

**Solutions:**
- Model errors as function of state
- Use local error correction
- Implement state-dependent integration method
- Consider hybrid/adaptive methods

## Code Structure

```python
# Main demonstration function
def main() -> None:
    """Run the Week 2 diagnostics demonstration."""
    
    # Step 1: Data generation
    # Step 2: Solver fitting
    # Step 3: Residual computation
    # Step 4: Diagnostic tests
    # Step 5: Report generation
    # Step 6: Visualization
    # Step 7: Summary table
    # Step 8: Interpretation
```

### Helper Functions

```python
def print_section(title: str) -> None:
    """Print formatted section header."""

def print_step(step_num: int, description: str) -> None:
    """Print formatted step header."""
```

## Requirements

```bash
pip install numpy matplotlib scipy statsmodels pysindy
```

Key dependencies:
- `numpy`: Numerical arrays and operations
- `matplotlib`: Plotting and visualization
- `scipy`: Statistical functions
- `statsmodels`: Time series analysis tools
- `pysindy`: Sparse identification of dynamics

## Integration with Week 1 & 2

This demo perfectly bridges:

**Week 1 (Solvers)** →
- RK45Solver implementation
- SINDy system identification
- ODE fitting and prediction

**Week 2 (Diagnostics)** →
- Statistical residual analysis
- DiagnosticEngine orchestration
- Report generation & visualization

**Pipeline:**
```
Synthetic Data → Solver Fit → Predictions → 
Residuals → Statistical Tests → Report & Plots
```

## Advanced Usage

### Modify Parameters

Edit demo to test different scenarios:

```python
# Change noise level
problem = exponential_decay(t_eval, x0=1.0, lambda_=0.5, noise_level=0.05)

# Change time span
t_eval = np.linspace(0, 10, 200)

# Different ODE system
problem = logistic_growth(t_eval, x0=0.1, r=1.0, K=1.0, noise_level=0.03)
```

### Disable Visualizations

Comment out visualization code if not needed:

```python
# In STEP 6, comment out:
# try:
#     fig = plot_diagnostics(t_eval, residuals, results)
#     ...
```

### Custom Solver Settings

```python
solver = RK45Solver(rtol=1e-8, atol=1e-12)  # Tighter tolerances
```

## Educational Value

This demo is designed for learning:

1. **Understand ODE Solving**: See how SINDy learns dynamics from data
2. **Learn Diagnostics**: Understand each statistical test
3. **Interpret Results**: Learn what test failures mean
4. **Explore Solutions**: See suggested improvements
5. **Visualize Analysis**: Understand diagnostic plots

## Output Interpretation

### Diagnostic Plot Explanation

The generated 2×2 plot shows:

**Top-left: Residuals Over Time**
- X-axis: Time
- Y-axis: Residual value
- Should be: Random scatter around zero
- Red line: Zero reference
- Warning: Trends indicate model bias

**Top-right: Autocorrelation Function**
- X-axis: Lag (time steps)
- Y-axis: Correlation coefficient
- Shaded area: Confidence interval
- Should be: Bars within shaded region
- Warning: Bars outside suggest correlation

**Bottom-left: Q-Q Plot**
- X-axis: Theoretical normal quantiles
- Y-axis: Sample quantiles
- Should be: Points near diagonal line
- Warning: S-curve indicates skewness
- Warning: Deviation at tails indicates heavy tails

**Bottom-right: Variance Trend**
- X-axis: Time
- Y-axis: Rolling standard deviation
- Should be: Flat horizontal line
- Warning: Increasing trend indicates heteroscedasticity
- Warning: Multiple trends indicate regime changes

## Performance Notes

- **Execution time**: 0.3-0.5 seconds
- **Plot generation**: Included in timing
- **Memory usage**: ~50 MB for all operations
- **PNG file size**: 280-300 KB at 150 dpi

## Troubleshooting

### ImportError: No module named 'statsmodels'
```bash
pip install statsmodels
```

### ImportError: No module named 'pysindy'
```bash
pip install pysindy
```

### Plot not saving
- Check write permissions in `examples/` directory
- Ensure matplotlib is properly installed
- Try different save format (change `.png` to `.pdf`)

### Solver not fitting
- Increase number of training points
- Try different ODE (logistic_growth instead of exponential_decay)
- Check if data has too much noise

## Next Steps

After running the demo:

1. **Examine the diagnostic plot** - Understand what each subplot shows
2. **Review test results** - Understand p-values and failure types
3. **Read recommendations** - See what improvements are suggested
4. **Try modifications** - Change noise level, time span, solver settings
5. **Experiment with different ODEs** - Test with logistic_growth, harmonic_oscillator
6. **Explore the codebase** - Look at DiagnosticEngine, statistical_tests
7. **Build custom diagnostics** - Use the framework for your own solvers

## Documentation References

For more information, see:
- `WEEK2_TESTING_SUMMARY.md` - Comprehensive testing documentation
- `examples/week1_demo.py` - Week 1 solver demonstration
- `ode_framework/diagnostics/` - Diagnostics module source code
- `ode_framework/tests/test_diagnostics.py` - 65 diagnostic tests

## Summary

The Week 2 diagnostics demo provides:
- ✅ Complete working example of diagnostics framework
- ✅ Educational explanation of each diagnostic test
- ✅ Practical guidance for interpreting results
- ✅ Real diagnostic plots and recommendations
- ✅ Bridge between Week 1 solvers and Week 2 diagnostics
- ✅ Copy-paste ready output for documentation
- ✅ Foundation for advanced custom diagnostics

It's the perfect starting point for understanding how to validate ODE solver solutions!
