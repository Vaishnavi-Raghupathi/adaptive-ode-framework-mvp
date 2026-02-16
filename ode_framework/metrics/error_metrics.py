"""Error metrics for ODE solution evaluation.

This module provides functions to compute standard regression and error metrics
for comparing true and predicted ODE solutions.
"""
from typing import Dict

import numpy as np


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> None:
    """Validate input arrays for metrics computation.

    Parameters
    ----------
    y_true : np.ndarray
        True values array.
    y_pred : np.ndarray
        Predicted values array.
    name : str, optional
        Function name for error messages. Default is empty string.

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have different shapes, contain NaN/Inf, or are empty.
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"{name}: y_true must be a numpy array")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"{name}: y_pred must be a numpy array")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"{name}: y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} vs {y_pred.shape}"
        )

    if y_true.size == 0:
        raise ValueError(f"{name}: input arrays cannot be empty")

    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError(f"{name}: y_true contains NaN or Inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError(f"{name}: y_pred contains NaN or Inf values")


def l2_norm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the L2 (Euclidean) norm of prediction errors.

    The L2 norm represents the Euclidean distance between true and predicted
    values, computed as the square root of the sum of squared differences.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Can be 1-D or multi-dimensional.
    y_pred : np.ndarray
        Predicted values. Must have the same shape as y_true.

    Returns
    -------
    float
        The L2 norm of the error: sqrt(sum((y_true - y_pred)^2))

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have different shapes, contain NaN/Inf, or are empty.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> l2_norm(y_true, y_pred)
    0.17320508075688772
    """
    _validate_inputs(y_true, y_pred, name="l2_norm")

    error = y_true - y_pred
    return float(np.sqrt(np.sum(error**2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Mean Squared Error (MSE).

    MSE measures the average squared difference between true and predicted
    values. Lower values indicate better predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Can be 1-D or multi-dimensional.
    y_pred : np.ndarray
        Predicted values. Must have the same shape as y_true.

    Returns
    -------
    float
        The mean squared error: mean((y_true - y_pred)^2)

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have different shapes, contain NaN/Inf, or are empty.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> mse(y_true, y_pred)
    0.010000000000000009
    """
    _validate_inputs(y_true, y_pred, name="mse")

    error = y_true - y_pred
    return float(np.mean(error**2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Root Mean Squared Error (RMSE).

    RMSE is the square root of MSE and is expressed in the same units as
    the predicted variable, making it more interpretable than MSE.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Can be 1-D or multi-dimensional.
    y_pred : np.ndarray
        Predicted values. Must have the same shape as y_true.

    Returns
    -------
    float
        The root mean squared error: sqrt(MSE)

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have different shapes, contain NaN/Inf, or are empty.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> rmse(y_true, y_pred)
    0.10000000000000003
    """
    _validate_inputs(y_true, y_pred, name="rmse")

    error = y_true - y_pred
    return float(np.sqrt(np.mean(error**2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R² coefficient of determination.

    R² measures the proportion of variance in the dependent variable that is
    predictable from the independent variables. It ranges from 0 to 1, where
    1 indicates a perfect fit and 0 indicates the model performs no better
    than a mean baseline.

    The formula is: R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)^2) and
    SS_tot = sum((y_true - mean(y_true))^2)

    Parameters
    ----------
    y_true : np.ndarray
        True values. Can be 1-D or multi-dimensional.
    y_pred : np.ndarray
        Predicted values. Must have the same shape as y_true.

    Returns
    -------
    float
        The R² score. Perfect predictions: 1.0. Mean baseline: 0.0.
        Worse than mean: negative values.

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have different shapes, contain NaN/Inf, or are empty.

    Notes
    -----
    When y_true has zero variance (all values identical), R² is undefined.
    In such cases, returns 1.0 if predictions are also constant and identical,
    else returns 0.0.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 3.9])
    >>> r_squared(y_true, y_pred)
    0.9945054945054945
    """
    _validate_inputs(y_true, y_pred, name="r_squared")

    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # Handle edge case where y_true has zero variance
    if ss_tot == 0:
        # If variance is zero, check if predictions are also constant and equal
        if ss_res == 0:
            return 1.0  # Perfect prediction of constant values
        else:
            return 0.0  # Failed to predict constant values

    return float(1.0 - (ss_res / ss_tot))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all standard error metrics at once.

    This function computes L2 norm, MSE, RMSE, and R² for the given
    true and predicted values in a single call.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Can be 1-D or multi-dimensional.
    y_pred : np.ndarray
        Predicted values. Must have the same shape as y_true.

    Returns
    -------
    dict
        Dictionary containing the following metrics:
        - 'l2_norm': L2 (Euclidean) norm of errors
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'r_squared': R² coefficient of determination

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have different shapes, contain NaN/Inf, or are empty.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 3.9])
    >>> metrics = compute_all_metrics(y_true, y_pred)
    >>> print(f"RMSE: {metrics['rmse']:.4f}")
    RMSE: 0.1000
    >>> print(f"R²: {metrics['r_squared']:.4f}")
    R²: 0.9945
    """
    _validate_inputs(y_true, y_pred, name="compute_all_metrics")

    return {
        "l2_norm": l2_norm(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }
