"""Test problems and benchmark cases for ODE solvers.

This module provides analytical ODE test cases with known solutions for
validating and benchmarking ODE solver implementations.
"""
from typing import Dict, Optional, Union

import numpy as np


def add_noise(
    data: np.ndarray, noise_level: float = 0.01, random_state: Optional[int] = None
) -> np.ndarray:
    """Add Gaussian noise to data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    noise_level : float, optional
        Standard deviation of noise as a fraction of data range.
        Default is 0.01 (1%).
    random_state : int, optional
        Seed for reproducibility. If None, the random state is not set.

    Returns
    -------
    np.ndarray
        Data with added Gaussian noise, same shape as input.

    Raises
    ------
    ValueError
        If noise_level is negative.
    """
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    if random_state is not None:
        np.random.seed(random_state)

    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return data.copy()

    noise = np.random.normal(0, noise_level * data_range, data.shape)
    return data + noise


def exponential_decay(
    t: np.ndarray, x0: float = 1.0, lambda_: float = 1.0, noise_level: float = 0.0
) -> Dict[str, Union[np.ndarray, float, Dict[str, float]]]:
    """Exponential decay problem.

    Solves the first-order linear ODE:
        dx/dt = -λ * x
    with initial condition x(0) = x₀.

    The analytical solution is:
        x(t) = x₀ * exp(-λ * t)

    This is a canonical test problem for validating ODE solvers on
    exponentially decaying systems.

    Parameters
    ----------
    t : np.ndarray
        Time points for evaluation. Shape: (n_samples,)
    x0 : float, optional
        Initial condition x(0). Default is 1.0.
    lambda_ : float, optional
        Decay rate parameter λ > 0. Default is 1.0.
    noise_level : float, optional
        Standard deviation of noise as fraction of data range.
        Default is 0.0 (no noise).

    Returns
    -------
    dict
        Dictionary containing:
        - 't': Time points (n_samples,)
        - 'x_exact': Exact solution (n_samples,) or (n_samples, 1)
        - 'x0': Initial condition scalar
        - 'params': Parameter dictionary {'lambda': lambda_}

    Raises
    ------
    ValueError
        If lambda_ is non-positive or noise_level is negative.

    Examples
    --------
    >>> t = np.linspace(0, 5, 100)
    >>> problem = exponential_decay(t, x0=1.0, lambda_=0.5)
    >>> print(problem['x_exact'].shape)
    (100, 1)
    """
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    x_exact = x0 * np.exp(-lambda_ * t)

    if noise_level > 0:
        x_exact = add_noise(x_exact, noise_level=noise_level)

    x_exact = x_exact.reshape(-1, 1)

    return {
        "t": t,
        "x_exact": x_exact,
        "x0": x0,
        "params": {"lambda": lambda_},
    }


def harmonic_oscillator(
    t: np.ndarray,
    x0: float = 1.0,
    v0: float = 0.0,
    omega: float = 1.0,
    noise_level: float = 0.0,
) -> Dict[str, Union[np.ndarray, float, Dict[str, float]]]:
    """Harmonic oscillator problem.

    Solves the second-order linear ODE:
        d²x/dt² = -ω² * x
    with initial conditions x(0) = x₀, dx/dt(0) = v₀.

    Converted to a first-order system with state vector [x, v]:
        dx/dt = v
        dv/dt = -ω² * x

    The analytical solution is:
        x(t) = x₀ * cos(ω*t) + (v₀/ω) * sin(ω*t)
        v(t) = -x₀ * ω * sin(ω*t) + v₀ * cos(ω*t)

    Parameters
    ----------
    t : np.ndarray
        Time points for evaluation. Shape: (n_samples,)
    x0 : float, optional
        Initial position x(0). Default is 1.0.
    v0 : float, optional
        Initial velocity v(0) = dx/dt(0). Default is 0.0.
    omega : float, optional
        Angular frequency ω > 0. Default is 1.0.
    noise_level : float, optional
        Standard deviation of noise as fraction of data range.
        Default is 0.0 (no noise).

    Returns
    -------
    dict
        Dictionary containing:
        - 't': Time points (n_samples,)
        - 'x_exact': State vector [x, v] at each time. Shape: (n_samples, 2)
        - 'x0': Initial state [x0, v0]
        - 'params': Parameter dictionary {'omega': omega}

    Raises
    ------
    ValueError
        If omega is non-positive or noise_level is negative.

    Examples
    --------
    >>> t = np.linspace(0, 2*np.pi, 100)
    >>> problem = harmonic_oscillator(t, x0=1.0, v0=0.0, omega=1.0)
    >>> print(problem['x_exact'].shape)
    (100, 2)
    """
    if omega <= 0:
        raise ValueError("omega must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)

    x_exact = np.column_stack([x, v])

    if noise_level > 0:
        x_exact = add_noise(x_exact, noise_level=noise_level)

    return {
        "t": t,
        "x_exact": x_exact,
        "x0": np.array([x0, v0]),
        "params": {"omega": omega},
    }


def logistic_growth(
    t: np.ndarray,
    x0: float = 0.1,
    r: float = 1.0,
    K: float = 1.0,
    noise_level: float = 0.0,
) -> Dict[str, Union[np.ndarray, float, Dict[str, float]]]:
    """Logistic growth problem.

    Solves the nonlinear first-order ODE:
        dx/dt = r * x * (1 - x/K)
    with initial condition x(0) = x₀.

    This model describes population growth with carrying capacity K.
    The growth rate r controls how quickly the population approaches K.

    The analytical solution is:
        x(t) = K / (1 + ((K - x₀) / x₀) * exp(-r*t))

    Parameters
    ----------
    t : np.ndarray
        Time points for evaluation. Shape: (n_samples,)
    x0 : float, optional
        Initial population x(0), where 0 < x₀ < K. Default is 0.1.
    r : float, optional
        Growth rate r > 0. Default is 1.0.
    K : float, optional
        Carrying capacity K > 0. Default is 1.0.
    noise_level : float, optional
        Standard deviation of noise as fraction of data range.
        Default is 0.0 (no noise).

    Returns
    -------
    dict
        Dictionary containing:
        - 't': Time points (n_samples,)
        - 'x_exact': Exact solution (n_samples, 1)
        - 'x0': Initial condition scalar
        - 'params': Parameter dictionary {'r': r, 'K': K}

    Raises
    ------
    ValueError
        If x0, r, or K are non-positive, or if x0 >= K, or if noise_level
        is negative.

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> problem = logistic_growth(t, x0=0.1, r=1.0, K=1.0)
    >>> print(problem['x_exact'].shape)
    (100, 1)
    """
    if x0 <= 0 or x0 >= K:
        raise ValueError("x0 must satisfy 0 < x0 < K")
    if r <= 0:
        raise ValueError("r must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    exponent = np.exp(-r * t)
    x_exact = K / (1 + ((K - x0) / x0) * exponent)

    if noise_level > 0:
        x_exact = add_noise(x_exact, noise_level=noise_level)

    x_exact = x_exact.reshape(-1, 1)

    return {
        "t": t,
        "x_exact": x_exact,
        "x0": x0,
        "params": {"r": r, "K": K},
    }
