# Unit Tests Quick Reference Guide

## Test File Location
`ode_framework/tests/test_diagnostics.py`

## Test Organization

```
test_diagnostics.py
├── Fixtures (lines 31-98)
│   ├── synthetic_iid_residuals()
│   ├── synthetic_heteroscedastic_residuals()
│   ├── synthetic_autocorrelated_residuals()
│   ├── synthetic_nonstationary_residuals()
│   └── time_array()
│
├── TestBreuschPagan (7 tests)
│   ├── test_bp_detects_heteroscedasticity
│   ├── test_bp_passes_homoscedastic
│   ├── test_bp_return_dict_structure
│   ├── test_bp_with_nan_residuals
│   ├── test_bp_with_nan_predictors
│   ├── test_bp_with_length_mismatch
│   └── test_bp_with_insufficient_data
│
├── TestLjungBox (10 tests)
│   ├── test_lb_detects_autocorrelation
│   ├── test_lb_passes_iid
│   ├── test_lb_return_dict_structure
│   ├── test_lb_with_nan_residuals
│   ├── test_lb_with_insufficient_data
│   ├── test_lb_with_invalid_lags
│   └── test_lb_different_lag_values [parametrized: 5, 10, 20, 30]
│
├── TestAugmentedDickeyFuller (6 tests)
│   ├── test_adf_detects_nonstationarity
│   ├── test_adf_passes_stationary
│   ├── test_adf_return_dict_structure
│   ├── test_adf_critical_values_structure
│   ├── test_adf_with_nan
│   └── test_adf_with_insufficient_data
│
├── TestStateDependence (12 tests)
│   ├── test_state_dep_detects_dependence
│   ├── test_state_dep_passes_independent
│   ├── test_state_dep_return_dict_structure
│   ├── test_state_dep_with_nan_residuals
│   ├── test_state_dep_with_nan_state
│   ├── test_state_dep_with_length_mismatch
│   ├── test_state_dep_with_insufficient_data
│   ├── test_state_dep_multidimensional_state
│   └── test_state_dep_various_dimensions [parametrized: 1, 2, 3, 5]
│
└── TestIntegration (3 tests)
    ├── test_all_tests_on_clean_data
    ├── test_all_tests_on_problematic_data
    └── test_all_tests_return_correct_structure
```

## Synthetic Data Generation

All synthetic data uses `seed=42` for reproducibility:

```python
# i.i.d. Gaussian (baseline - should pass all tests)
synthetic_iid_residuals
→ 100 points of N(0,1) noise

# Heteroscedastic (variance increases with time)
synthetic_heteroscedastic_residuals
→ time:      linspace(0, 1, 100)
→ residuals: noise × (1 + 2*time)
→ multiplier ranges from 1 to 3

# Autocorrelated (AR(1) process)
synthetic_autocorrelated_residuals
→ AR(1) with coefficient φ = 0.7
→ Strong autocorrelation detected

# Non-stationary (Random walk)
synthetic_nonstationary_residuals
→ cumsum(noise)
→ Unit root, clearly non-stationary

# Time dimension
time_array
→ linspace(0, 10, 100)
```

## Running Specific Tests

```bash
# Run all diagnostic tests
pytest ode_framework/tests/test_diagnostics.py -v

# Run specific test class
pytest ode_framework/tests/test_diagnostics.py::TestBreuschPagan -v

# Run specific test
pytest ode_framework/tests/test_diagnostics.py::TestBreuschPagan::test_bp_detects_heteroscedasticity -v

# Run with short summary
pytest ode_framework/tests/test_diagnostics.py --tb=short -q

# Run only parametrized tests
pytest ode_framework/tests/test_diagnostics.py::TestLjungBox::test_lb_different_lag_values -v

# Run integration tests only
pytest ode_framework/tests/test_diagnostics.py::TestIntegration -v

# Run with pattern matching
pytest ode_framework/tests/test_diagnostics.py -k "detects" -v
pytest ode_framework/tests/test_diagnostics.py -k "nan" -v
pytest ode_framework/tests/test_diagnostics.py -k "structure" -v
```

## Key Test Patterns

### Positive Case Pattern
```python
def test_<function>_detects_<issue>(self):
    """Should detect actual issue in data"""
    # Create data with known issue
    result = <function>(problematic_data)
    
    # Assert detection
    assert result['<issue_flag>'] == True
    assert result['p_value'] < 0.05
```

### Negative Case Pattern
```python
def test_<function>_passes_<condition>(self):
    """Should pass when issue not present"""
    # Use clean data
    result = <function>(iid_residuals)
    
    # Assert no detection
    assert result['<issue_flag>'] == False
    assert result['p_value'] > 0.05
```

### Input Validation Pattern
```python
def test_<function>_with_<invalid_input>(self):
    """Should reject invalid input"""
    with pytest.raises(ValueError, match="<expected_error>"):
        <function>(invalid_data)
```

### Structure Validation Pattern
```python
def test_<function>_return_dict_structure(self):
    """Should return dict with correct keys and types"""
    result = <function>(clean_data)
    
    # Check keys
    expected_keys = {'key1', 'key2', ...}
    assert set(result.keys()) == expected_keys
    
    # Check types
    assert isinstance(result['key1'], expected_type)
```

## Test Statistics at a Glance

| Aspect | Count |
|--------|-------|
| Total test cases | 38 |
| Test classes | 5 |
| Fixtures | 5 |
| Parametrized parameters | 8 (4 lags + 4 dimensions) |
| Input validation tests | 14 |
| Positive detection tests | 4 |
| Negative detection tests | 4 |
| Structure validation tests | 5 |
| Integration tests | 3 |
| **Pass rate** | **100%** ✅ |

## Common Test Assertions

```python
# Boolean assertions (use == not 'is' for numpy.bool_)
assert result['heteroscedastic'] == True    # ✓ Correct
assert result['heteroscedastic'] is True    # ✗ May fail

# Dict structure validation
assert set(result.keys()) == expected_keys  # ✓ Check all keys present

# Type checking
assert isinstance(result['value'], (int, float, np.number))

# P-value checks
assert result['p_value'] < 0.05  # Significant
assert result['p_value'] > 0.05  # Not significant

# Error raising
with pytest.raises(ValueError, match="NaN"):
    function_under_test(bad_data)

# Parametrized test access
@pytest.mark.parametrize("param", [1, 2, 3])
def test_function(self, param):
    # param is 1, then 2, then 3
```

## Debugging Failed Tests

If a test fails, check:

1. **Fixture data**: Print the fixture to verify correct generation
   ```python
   def test_debug(self, synthetic_heteroscedastic_residuals):
       time, residuals = synthetic_heteroscedastic_residuals
       print(f"Time range: {time.min()}-{time.max()}")
       print(f"Residuals std range: {residuals.std()}")
   ```

2. **Actual vs expected values**: Run individually
   ```bash
   pytest test_diagnostics.py::TestBreuschPagan::test_bp_detects_heteroscedasticity -vv -s
   ```

3. **P-values**: May vary slightly due to random seed - adjust thresholds if needed
   ```python
   # More relaxed threshold for synthetic data
   assert result['p_value'] < 0.1  # Instead of 0.05
   ```

4. **Numpy scalar types**: Remember results return numpy types
   ```python
   # Check actual type
   print(type(result['heteroscedastic']))  # <class 'numpy.bool_'>
   
   # Use == for comparison, not 'is'
   assert result['heteroscedastic'] == True
   ```

## Test Coverage Visualization

```
statistical_tests.py
├── breusch_pagan_test ━━━━━━━━━━━━ 7 tests (100%)
│   ├─ Positive: Heteroscedasticity detected
│   ├─ Negative: Homoscedasticity passes
│   └─ Validation: NaN, mismatches, insufficient data
│
├── ljung_box_test ━━━━━━━━━━━━━━ 10 tests (100%)
│   ├─ Positive: Autocorrelation detected
│   ├─ Negative: IID passes
│   ├─ Parametrized: 4 lag values
│   └─ Validation: NaN, mismatches, invalid lags
│
├── augmented_dickey_fuller_test ━━ 6 tests (100%)
│   ├─ Positive: Non-stationarity (random walk)
│   ├─ Negative: Stationarity passes
│   ├─ Critical values: 1%, 5%, 10% verified
│   └─ Validation: NaN, insufficient data
│
├── state_dependence_test ━━━━━━━ 12 tests (100%)
│   ├─ Positive: Linear correlation detected
│   ├─ Negative: Independence passes
│   ├─ Multi-dimensional: 4 state dimensions
│   └─ Validation: NaN, mismatches, insufficient data
│
└── Integration ━━━━━━━━━━━━━━━━━ 3 tests (100%)
    ├─ All tests on clean data
    ├─ All tests on problematic data
    └─ Return structure consistency
```

## Tips for Adding New Tests

1. **Use existing fixtures**: Reuse synthetic data generators
2. **Follow naming**: `test_<function>_<aspect>` (e.g., `test_bp_with_nan_residuals`)
3. **Use descriptive docstrings**: Explain what the test verifies
4. **Test both positive and negative**: Create complementary test pairs
5. **Parametrize repeated tests**: Use `@pytest.mark.parametrize` for multiple values
6. **Check error messages**: Use `match=` to verify exception messages
7. **Keep tests focused**: One assertion concept per test (though multiple assertions of same concept okay)
8. **Use fixtures consistently**: Call via function parameters, not by name

## Integration with CI/CD

The test suite integrates seamlessly with:

```bash
# Full project test suite
pytest ode_framework/tests/ -q

# With coverage reporting
pytest ode_framework/tests/ --cov=ode_framework.diagnostics.statistical_tests

# Pre-commit hook
pytest ode_framework/tests/test_diagnostics.py --tb=short -q

# GitHub Actions (example)
pytest ode_framework/tests/test_diagnostics.py -v --tb=short
```

## Expected Output

```
==================== 38 passed in 0.66s ====================

PASSED ode_framework/tests/test_diagnostics.py::TestBreuschPagan::...
PASSED ode_framework/tests/test_diagnostics.py::TestLjungBox::...
PASSED ode_framework/tests/test_diagnostics.py::TestAugmentedDickeyFuller::...
PASSED ode_framework/tests/test_diagnostics.py::TestStateDependence::...
PASSED ode_framework/tests/test_diagnostics.py::TestIntegration::...
```

---

For more details, see `TEST_COVERAGE_SUMMARY.md`
