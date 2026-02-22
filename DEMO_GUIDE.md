# Complete Demo Guide - Week 1 & Week 2

## Quick Start

### Run Week 1 Demo (ODE Solvers)
```bash
cd examples
python week1_demo.py
```

**Output:**
- Console output showing solver fitting and predictions
- Generated plot: `week1_demo.png`
- Topics: Solver comparison, metrics, convergence

**Duration:** ~10-15 seconds

### Run Week 2 Demo (Diagnostics)
```bash
cd examples
python week2_demo.py
```

**Output:**
- Console output with 8-step diagnostic pipeline
- Generated plot: `week2_diagnostics.png`
- Topics: Statistical tests, failure detection, recommendations

**Duration:** ~0.5 seconds

---

## Demo Comparison

| Aspect | Week 1 | Week 2 |
|--------|--------|--------|
| **Focus** | Solving ODEs | Validating Solutions |
| **Input** | Test problem + solver type | Residuals + time points |
| **Process** | Fit → Predict → Evaluate | Analyze → Test → Report |
| **Output** | Solution plots + metrics | Diagnostic plots + recommendations |
| **Tests** | RMSE, R², convergence | Statistical tests (4 types) |
| **Duration** | 10-15 seconds | <1 second |
| **Use Case** | Choose best solver | Improve solver quality |

---

## Week 1 Demo: `examples/week1_demo.py`

### What It Shows

1. **Generate synthetic data** (exponential decay)
2. **Fit solvers** (RK4, RK45)
3. **Compare performance** (RMSE, R², etc.)
4. **Visualize solutions** (exact vs. predicted)

### Key Output

```
[STEP 1] Generate analytical exponential decay data...
  ✓ Generated 100 training points
  ✓ Generated 200 evaluation points

[STEP 2] Fit RK4Solver to noisy training data...
  ✓ Solver fitted successfully

[STEP 3] Fit RK45Solver to noisy training data...
  ✓ Solver fitted successfully

[STEP 4] Evaluate metrics on fine evaluation grid...
  RK4Solver:
    RMSE: 0.0089
    R²: 0.9989
  RK45Solver:
    RMSE: 0.0083
    R²: 0.9991

[STEP 5] Plot results and save to disk...
  ✓ Plot saved to examples/week1_demo.png
```

### When to Use

- ✅ Learning how solvers work
- ✅ Comparing different ODE methods
- ✅ Understanding SINDy system identification
- ✅ Evaluating solver accuracy metrics

---

## Week 2 Demo: `examples/week2_demo.py`

### What It Shows

1. **Generate synthetic ODE data** (3% noise, 150 points)
2. **Fit solver** (RK45 with SINDy)
3. **Compute residuals** (prediction errors)
4. **Run 4 statistical tests**:
   - Breusch-Pagan (heteroscedasticity)
   - Ljung-Box (autocorrelation)
   - ADF (stationarity)
   - State-Dependence
5. **Generate report** (test results + recommendations)
6. **Create visualizations** (2×2 diagnostic plot)
7. **Interpret results** (what failures mean)
8. **Suggest improvements** (how to fix issues)

### Key Output

```
[STEP 1] Generate synthetic exponential decay data with noise
  • Time points: 150 (0 to 5 seconds)
  • Noise level: 3%
  ✓ Synthetic data generated successfully

[STEP 2] Fit RK45Solver to noisy data using SINDy
  ✓ Solver fitted successfully

[STEP 3] Generate predictions and compute residuals
  • Residuals mean: -0.0547
  • Residuals std: 0.068
  • RMSE: 0.0873

[STEP 4] Run comprehensive statistical diagnostic tests
  ✓ All diagnostic tests completed successfully

  Test Results:
  - Breusch-Pagan Test (Heteroscedasticity)        FAILED ✗
  - Ljung-Box Test (Autocorrelation)               FAILED ✗
  - ADF Test (Stationarity)                        FAILED ✗
  - State-Dependence Test                          FAILED ✗

[STEP 5] Generate diagnostic report
  Detected Issues: heteroscedastic, autocorrelated, 
                   nonstationary, state_dependent
  
  Recommendation: Consider regime-switching model or 
                  time-varying parameters

[STEP 6] Create diagnostic visualizations
  ✓ Diagnostic plot saved
  • Location: examples/week2_diagnostics.png
  • File size: 282 KB

[STEP 7] Formatted diagnostic summary table
  DIAGNOSTIC TEST RESULTS
  Test Name                    Detected  P-Value
  Breusch-Pagan              YES       0.000000  FAILED ✗
  Ljung-Box                  YES       0.000000  FAILED ✗
  ADF Test                   YES       0.998023  FAILED ✗
  State-Dependence           YES       0.000000  FAILED ✗

[STEP 8] Interpretation and next steps
  ⚠ Issues Detected (4 failures)
  
  • Heteroscedasticity detected → Consider SDE formulation
  • Autocorrelation detected → Add missing terms/complexity
  • Non-stationarity detected → Check for regime changes
  • State-dependence detected → Use adaptive error correction
```

### Diagnostic Plot (2×2 Grid)

**Top-left: Residuals over time**
- Scatter plot with zero reference line
- Should be: Random around zero
- Warning: Systematic trends indicate bias

**Top-right: Autocorrelation (ACF)**
- Bar plot with confidence interval
- Should be: Bars within shaded region
- Warning: Bars outside suggest correlation

**Bottom-left: Q-Q plot**
- Normal probability plot
- Should be: Points near diagonal line
- Warning: Deviation indicates non-normality

**Bottom-right: Variance trend**
- Rolling standard deviation
- Should be: Flat horizontal line
- Warning: Trending indicates heteroscedasticity

### When to Use

- ✅ Validating solver quality
- ✅ Detecting systematic model errors
- ✅ Understanding residual structure
- ✅ Guiding solver improvements
- ✅ Learning diagnostic methods

---

## Complete Pipeline: Week 1 → Week 2

```
Real-world ODE Problem
        ↓
[WEEK 1] Fit Solver with SINDy
        ↓
Generate Predictions
        ↓
Compute Residuals
        ↓
[WEEK 2] Run Diagnostics
        ↓
Statistical Tests (4 types)
        ↓
Identify Issues
        ↓
Get Recommendations
        ↓
Improve Solver / Method
        ↓
[Loop back or accept solution]
```

### Example Use Case: Building a Production ODE Solver

1. **Generate realistic test data** with noise
2. **Fit multiple solvers** (Week 1 capabilities)
3. **Compare performance** (metrics, speed)
4. **Select best solver** (e.g., RK45)
5. **Run diagnostics** (Week 2 framework)
6. **Identify issues** (e.g., autocorrelation, state-dependence)
7. **Improve solver** (e.g., add SDE formulation, Neural ODE)
8. **Re-run diagnostics** (validate improvement)
9. **Deploy to production** (confident in quality)

---

## File Locations

```
examples/
├── week1_demo.py              ← Solver demonstration
├── week1_demo.png             ← Generated solver plot
├── week2_demo.py              ← Diagnostics demonstration
├── week2_diagnostics.png      ← Generated diagnostic plot
├── WEEK2_DEMO_README.md       ← Detailed diagnostics guide
└── README.md                  ← Project overview

ode_framework/
├── solvers/
│   └── classical.py           ← RK4Solver, RK45Solver
├── diagnostics/
│   ├── statistical_tests.py   ← 4 statistical tests
│   ├── diagnostic_engine.py   ← Orchestration engine
│   └── visualizations.py      ← Plotting functions
├── utils/
│   └── test_problems.py       ← exponential_decay, etc.
└── tests/
    └── test_diagnostics.py    ← 65 comprehensive tests
```

---

## Common Tasks

### Task: Test a new solver

```bash
# 1. Run Week 1 demo to establish baseline
python examples/week1_demo.py

# 2. Create custom solver in ode_framework/solvers/
# 3. Update week1_demo.py to include your solver
# 4. Re-run to compare performance
```

### Task: Improve a solver that fails diagnostics

```bash
# 1. Run Week 2 demo to see what fails
python examples/week2_demo.py

# 2. Review the failure type:
#    - Autocorrelation → Add complexity
#    - Heteroscedasticity → Increase tolerance
#    - Non-stationarity → Use regime model
#    - State-dependence → Use adaptive method

# 3. Modify solver accordingly

# 4. Re-run to verify improvement
python examples/week2_demo.py
```

### Task: Create a custom diagnostic test

```python
# In ode_framework/diagnostics/statistical_tests.py
def my_custom_test(residuals, threshold=0.05):
    """My custom diagnostic test."""
    # Your test implementation
    return {"p_value": p_val, "my_metric": result}

# Add to DiagnosticEngine.run_diagnostics()
# Create test in ode_framework/tests/test_diagnostics.py
# Run: pytest ode_framework/tests/test_diagnostics.py -v
```

---

## Requirements

```bash
# Install all dependencies
pip install numpy scipy matplotlib statsmodels pysindy pytest

# Or install from requirements
pip install -r requirements.txt
```

### Optional for performance
```bash
pip install numba numexpr
```

---

## Troubleshooting

### Demo runs but produces no output
- Check that `matplotlib` backend is configured
- Verify `pandas` and `statsmodels` are installed
- Try: `python -c "import matplotlib; print(matplotlib.get_backend())"`

### Plots not saving
- Ensure `examples/` directory is writable
- Check disk space available
- Try running with admin/sudo if permission denied

### ImportError for pysindy
- Install: `pip install pysindy`
- Note: Requires Python 3.7+

### SINDy warnings during fit
- These are normal and expected
- Suppress with: `warnings.filterwarnings('ignore')`

### Slow execution
- Reduce number of time points
- Use RK4 instead of RK45 (faster, less accurate)
- Increase solver tolerance (less accurate, faster)

---

## Learning Path

### Beginner (Start Here)
1. Read this guide
2. Run `week1_demo.py`
3. Run `week2_demo.py`
4. Review generated plots
5. Read output interpretation guide

### Intermediate
1. Modify demo parameters (noise level, time span)
2. Try different test problems (logistic, harmonic)
3. Read WEEK2_DEMO_README.md in detail
4. Explore `ode_framework/diagnostics/` source
5. Review statistical test functions

### Advanced
1. Create custom solvers
2. Implement custom diagnostic tests
3. Modify demo to test your ideas
4. Read WEEK2_TESTING_SUMMARY.md (65 tests)
5. Contribute improvements to framework

---

## Key Insights

### From Week 1
- Multiple solvers have different trade-offs
- SINDy can learn ODEs from noisy data
- RK45 is generally better than RK4 (adaptive)
- Metrics matter: RMSE, R², convergence

### From Week 2
- Residuals tell important stories
- Statistical tests are automated validators
- Different failures need different solutions
- Diagnostics guide model improvements
- Visual inspection complements statistics

---

## Quick Reference: Diagnostic Failures

| Test | Failure | Cause | Solution |
|------|---------|-------|----------|
| **Heteroscedasticity** | Variance changes | Model incomplete | SDE, adaptive |
| **Autocorrelation** | Residuals correlated | Missing structure | Add terms, Neural ODE |
| **Non-stationarity** | Mean/variance shift | Regime changes | Switching model, time-varying |
| **State-dependence** | Error depends on state | State-specific error | Adaptive error, local models |

---

## Next Steps

After running the demos:

1. **Review the test suite**: `pytest ode_framework/tests/test_diagnostics.py -v`
2. **Explore the source code**: `ode_framework/diagnostics/`
3. **Modify and experiment**: Edit demos to test ideas
4. **Create custom tests**: Implement your own diagnostics
5. **Build real applications**: Use framework for your problems

---

## Further Reading

- `WEEK2_TESTING_SUMMARY.md` - Comprehensive testing documentation
- `examples/WEEK2_DEMO_README.md` - Detailed diagnostics guide
- `ode_framework/diagnostics/README.md` - Module documentation (if available)
- Source code docstrings - Inline documentation in Python files

---

## Summary

The demo framework demonstrates:
- ✅ How to solve ODEs with multiple solvers (Week 1)
- ✅ How to validate solutions with diagnostics (Week 2)
- ✅ How to interpret diagnostic results
- ✅ How to guide solver improvements
- ✅ Complete pipeline from problem to production

**Start with:** `python examples/week2_demo.py`

**Learn more:** See `examples/WEEK2_DEMO_README.md`

**Test everything:** `pytest ode_framework/tests/ -v`

Happy exploring! 🚀
