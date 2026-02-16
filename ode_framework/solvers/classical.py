"""Classical ODE solver implementations.

This module provides concrete implementations of classical ODE solvers using
SINDy for system identification and scipy for numerical integration.
"""
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .base import BaseSolver

# Optional SINDy import - will be required for fit() method
try:
    from pysindy import SINDy
    HAS_SINDY = True
except ImportError:
    HAS_SINDY = False

if TYPE_CHECKING:
    from pysindy import SINDy


class RK4Solver(BaseSolver):
    """Fixed-step Runge-Kutta 4th order ODE solver with SINDy identification.

    This solver uses SINDy (Sparse Identification of Nonlinear Dynamics) to
    identify the ODE governing the dynamics from training data, then solves
    the identified ODE using scipy's dopri5 integrator (which implements
    Runge-Kutta 4th/5th order methods).

    Attributes
    ----------
    ode_function : callable, optional
        The learned ODE function dx/dt = f(t, x). Populated after fit().
    sindy_model : SINDy, optional
        The trained SINDy model. Stored after fit() for inspection.
    t_data : np.ndarray, optional
        Training time data. Stored for reference.
    x_data : np.ndarray, optional
        Training state data. Stored for reference.
    _is_fitted : bool
        Flag indicating whether the solver has been fitted.

    Notes
    -----
    Requires pysindy to be installed for the fit() method.
    """

    def __init__(self, sindy_model: Optional["SINDy"] = None) -> None:
        """Initialize the RK4Solver.

        Parameters
        ----------
        sindy_model : SINDy, optional
            Pre-configured SINDy model. If None, a default SINDy() instance
            will be created during fit(). Default is None.
        """
        if sindy_model is not None and not HAS_SINDY:
            raise ImportError("pysindy is required but not installed.")

        self.sindy_model = sindy_model
        self.ode_function: Optional[Callable] = None
        self.t_data: Optional[np.ndarray] = None
        self.x_data: Optional[np.ndarray] = None
        self._is_fitted = False
        self._is_univariate = False  # Track if original data was 1D

    def _validate_inputs(
        self, t_data: np.ndarray, x_data: np.ndarray, name: str = "input"
    ) -> None:
        """Validate input arrays for shape and NaN values.

        Parameters
        ----------
        t_data : np.ndarray
            Time points array.
        x_data : np.ndarray
            State values array.
        name : str, optional
            Name for error messages. Default is "input".

        Raises
        ------
        ValueError
            If arrays contain NaN/Inf values or have incompatible shapes.
        """
        if not isinstance(t_data, np.ndarray):
            raise TypeError(f"{name} t_data must be a numpy array")
        if not isinstance(x_data, np.ndarray):
            raise TypeError(f"{name} x_data must be a numpy array")

        if np.any(np.isnan(t_data)) or np.any(np.isinf(t_data)):
            raise ValueError(f"{name} t_data contains NaN or Inf values")
        if np.any(np.isnan(x_data)) or np.any(np.isinf(x_data)):
            raise ValueError(f"{name} x_data contains NaN or Inf values")

        if t_data.ndim != 1:
            raise ValueError(f"{name} t_data must be 1-dimensional, got shape {t_data.shape}")
        if x_data.ndim not in [1, 2]:
            raise ValueError(
                f"{name} x_data must be 1-D or 2-D, got shape {x_data.shape}"
            )

        if len(t_data) != x_data.shape[0]:
            raise ValueError(
                f"{name} t_data and x_data have incompatible lengths: "
                f"{len(t_data)} vs {x_data.shape[0]}"
            )

    def fit(self, t_data: np.ndarray, x_data: np.ndarray) -> None:
        """Fit the solver by identifying ODE dynamics using SINDy.

        This method uses SINDy to learn a sparse ODE model from the provided
        time series data. The learned model is stored internally for later
        predictions.

        Parameters
        ----------
        t_data : np.ndarray
            Time points of training data. Shape: (n_samples,)
        x_data : np.ndarray
            State values at training times. Shape: (n_samples, n_states)
            or (n_samples,) for univariate systems.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If pysindy is not installed.
        ValueError
            If input arrays are invalid (NaN, shape mismatch, etc.).
        RuntimeError
            If SINDy fitting fails to identify a model.
        """
        if not HAS_SINDY:
            raise ImportError(
                "pysindy is required for RK4Solver.fit(). "
                "Install it with: pip install pysindy"
            )

        self._validate_inputs(t_data, x_data, name="fit")

        # Track if input was univariate
        self._is_univariate = x_data.ndim == 1

        # Ensure x_data is 2D
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)

        # Create SINDy model if not provided
        if self.sindy_model is None:
            self.sindy_model = SINDy()

        # Fit the SINDy model
        try:
            self.sindy_model.fit(x_data, t=t_data)
        except Exception as e:
            raise RuntimeError(f"SINDy model fitting failed: {str(e)}") from e

        # Create ODE function from the learned model
        def ode_func(t: float, x: np.ndarray) -> np.ndarray:
            """ODE function dx/dt = f(t, x) learned by SINDy."""
            return self.sindy_model.predict(x.reshape(1, -1)).flatten()

        self.ode_function = ode_func
        self.t_data = t_data.copy()
        self.x_data = x_data.copy()
        self._is_fitted = True

    def predict(self, t_eval: np.ndarray) -> np.ndarray:
        """Predict state values at given times using the fitted ODE.

        Solves the learned ODE using scipy's solve_ivp with dopri5 method
        starting from the initial condition in the training data.

        Parameters
        ----------
        t_eval : np.ndarray
            Time points for evaluation. Shape: (n_eval,)

        Returns
        -------
        np.ndarray
            Predicted state values. Shape: (n_eval, n_states) or (n_eval,)
            for univariate systems.

        Raises
        ------
        RuntimeError
            If the solver has not been fitted before prediction.
        ValueError
            If t_eval is invalid.
        """
        if not self._is_fitted:
            raise RuntimeError("Solver must be fitted before calling predict()")

        if not isinstance(t_eval, np.ndarray):
            raise TypeError("t_eval must be a numpy array")
        if np.any(np.isnan(t_eval)) or np.any(np.isinf(t_eval)):
            raise ValueError("t_eval contains NaN or Inf values")
        if t_eval.ndim != 1:
            raise ValueError(f"t_eval must be 1-dimensional, got shape {t_eval.shape}")

        # Get initial condition from training data
        x0 = self.x_data[0]
        t_span = (t_eval[0], t_eval[-1])

        # Solve ODE using solve_ivp with dopri5 (Runge-Kutta 4th/5th order)
        try:
            solution = solve_ivp(
                self.ode_function,
                t_span,
                x0,
                method="RK45",
                t_eval=t_eval,
                rtol=1e-3,
                atol=1e-6,
                dense_output=False,
            )

            if not solution.success:
                raise RuntimeError(
                    f"RK4 integration failed: {solution.message}"
                )

            result = solution.y.T
            # Return 1D array if input was univariate
            if self._is_univariate:
                result = result.flatten()
            return result
        except Exception as e:
            raise RuntimeError(f"ODE integration failed: {str(e)}") from e

    def compute_residuals(self, t_data: np.ndarray, x_data: np.ndarray) -> np.ndarray:
        """Compute residuals between actual and predicted values.

        Parameters
        ----------
        t_data : np.ndarray
            Time points. Shape: (n_samples,)
        x_data : np.ndarray
            Actual state values. Shape: (n_samples, n_states) or (n_samples,)

        Returns
        -------
        np.ndarray
            Residuals (x_data - predictions). Same shape as x_data.

        Raises
        ------
        RuntimeError
            If the solver has not been fitted.
        ValueError
            If input arrays are invalid.
        """
        if not self._is_fitted:
            raise RuntimeError("Solver must be fitted before computing residuals")

        self._validate_inputs(t_data, x_data, name="compute_residuals")

        predictions = self.predict(t_data)
        return x_data - predictions


class RK45Solver(BaseSolver):
    """Adaptive Runge-Kutta-Fehlberg ODE solver with SINDy identification.

    This solver uses SINDy for system identification and scipy's solve_ivp
    with the 'RK45' method (Runge-Kutta 4th/5th order Fehlberg) for adaptive
    time-stepping ODE integration.

    Attributes
    ----------
    ode_function : callable, optional
        The learned ODE function dx/dt = f(t, x). Populated after fit().
    sindy_model : SINDy, optional
        The trained SINDy model. Stored after fit() for inspection.
    t_data : np.ndarray, optional
        Training time data. Stored for reference.
    x_data : np.ndarray, optional
        Training state data. Stored for reference.
    _is_fitted : bool
        Flag indicating whether the solver has been fitted.
    rtol : float
        Relative tolerance for RK45 integration.
    atol : float
        Absolute tolerance for RK45 integration.

    Notes
    -----
    Requires pysindy to be installed for the fit() method.
    The RK45 method provides adaptive time-stepping for improved accuracy.
    """

    def __init__(
        self,
        sindy_model: Optional["SINDy"] = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ) -> None:
        """Initialize the RK45Solver.

        Parameters
        ----------
        sindy_model : SINDy, optional
            Pre-configured SINDy model. If None, a default SINDy() instance
            will be created during fit(). Default is None.
        rtol : float, optional
            Relative tolerance for integration. Default is 1e-3.
        atol : float, optional
            Absolute tolerance for integration. Default is 1e-6.
        """
        if sindy_model is not None and not HAS_SINDY:
            raise ImportError("pysindy is required but not installed.")

        self.sindy_model = sindy_model
        self.ode_function: Optional[Callable] = None
        self.t_data: Optional[np.ndarray] = None
        self.x_data: Optional[np.ndarray] = None
        self._is_fitted = False
        self._is_univariate = False  # Track if original data was 1D
        self.rtol = rtol
        self.atol = atol

    def _validate_inputs(
        self, t_data: np.ndarray, x_data: np.ndarray, name: str = "input"
    ) -> None:
        """Validate input arrays for shape and NaN values.

        Parameters
        ----------
        t_data : np.ndarray
            Time points array.
        x_data : np.ndarray
            State values array.
        name : str, optional
            Name for error messages. Default is "input".

        Raises
        ------
        ValueError
            If arrays contain NaN/Inf values or have incompatible shapes.
        """
        if not isinstance(t_data, np.ndarray):
            raise TypeError(f"{name} t_data must be a numpy array")
        if not isinstance(x_data, np.ndarray):
            raise TypeError(f"{name} x_data must be a numpy array")

        if np.any(np.isnan(t_data)) or np.any(np.isinf(t_data)):
            raise ValueError(f"{name} t_data contains NaN or Inf values")
        if np.any(np.isnan(x_data)) or np.any(np.isinf(x_data)):
            raise ValueError(f"{name} x_data contains NaN or Inf values")

        if t_data.ndim != 1:
            raise ValueError(f"{name} t_data must be 1-dimensional, got shape {t_data.shape}")
        if x_data.ndim not in [1, 2]:
            raise ValueError(
                f"{name} x_data must be 1-D or 2-D, got shape {x_data.shape}"
            )

        if len(t_data) != x_data.shape[0]:
            raise ValueError(
                f"{name} t_data and x_data have incompatible lengths: "
                f"{len(t_data)} vs {x_data.shape[0]}"
            )

    def fit(self, t_data: np.ndarray, x_data: np.ndarray) -> None:
        """Fit the solver by identifying ODE dynamics using SINDy.

        This method uses SINDy to learn a sparse ODE model from the provided
        time series data. The learned model is stored internally for later
        predictions with adaptive time-stepping.

        Parameters
        ----------
        t_data : np.ndarray
            Time points of training data. Shape: (n_samples,)
        x_data : np.ndarray
            State values at training times. Shape: (n_samples, n_states)
            or (n_samples,) for univariate systems.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If pysindy is not installed.
        ValueError
            If input arrays are invalid (NaN, shape mismatch, etc.).
        RuntimeError
            If SINDy fitting fails or convergence issues occur.
        """
        if not HAS_SINDY:
            raise ImportError(
                "pysindy is required for RK45Solver.fit(). "
                "Install it with: pip install pysindy"
            )

        self._validate_inputs(t_data, x_data, name="fit")

        # Track if input was univariate
        self._is_univariate = x_data.ndim == 1

        # Ensure x_data is 2D
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)

        # Create SINDy model if not provided
        if self.sindy_model is None:
            self.sindy_model = SINDy()

        # Fit the SINDy model
        try:
            self.sindy_model.fit(x_data, t=t_data)
        except Exception as e:
            raise RuntimeError(f"SINDy model fitting failed: {str(e)}") from e

        # Create ODE function from the learned model
        def ode_func(t: float, x: np.ndarray) -> np.ndarray:
            """ODE function dx/dt = f(t, x) learned by SINDy."""
            return self.sindy_model.predict(x.reshape(1, -1)).flatten()

        self.ode_function = ode_func
        self.t_data = t_data.copy()
        self.x_data = x_data.copy()
        self._is_fitted = True

    def predict(self, t_eval: np.ndarray) -> np.ndarray:
        """Predict state values at given times using adaptive RK45 integration.

        Solves the learned ODE using scipy's solve_ivp with the RK45 method,
        which provides adaptive time-stepping for improved accuracy and
        efficiency.

        Parameters
        ----------
        t_eval : np.ndarray
            Time points for evaluation. Shape: (n_eval,)

        Returns
        -------
        np.ndarray
            Predicted state values. Shape: (n_eval, n_states) or (n_eval,)
            for univariate systems.

        Raises
        ------
        RuntimeError
            If the solver has not been fitted or integration fails.
        ValueError
            If t_eval is invalid.
        """
        if not self._is_fitted:
            raise RuntimeError("Solver must be fitted before calling predict()")

        if not isinstance(t_eval, np.ndarray):
            raise TypeError("t_eval must be a numpy array")
        if np.any(np.isnan(t_eval)) or np.any(np.isinf(t_eval)):
            raise ValueError("t_eval contains NaN or Inf values")
        if t_eval.ndim != 1:
            raise ValueError(f"t_eval must be 1-dimensional, got shape {t_eval.shape}")

        # Get initial condition from training data
        x0 = self.x_data[0]
        t_span = (t_eval[0], t_eval[-1])

        # Solve ODE using RK45 with adaptive time-stepping
        try:
            solution = solve_ivp(
                self.ode_function,
                t_span,
                x0,
                method="RK45",
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=False,
            )

            if not solution.success:
                raise RuntimeError(
                    f"RK45 integration failed: {solution.message}"
                )

            result = solution.y.T
            # Return 1D array if input was univariate
            if self._is_univariate:
                result = result.flatten()
            return result
            # Return 1D array if input was univariate
            if self._is_univariate:
                result = result.flatten()
            return result
        except Exception as e:
            raise RuntimeError(f"ODE integration failed: {str(e)}") from e

    def compute_residuals(self, t_data: np.ndarray, x_data: np.ndarray) -> np.ndarray:
        """Compute residuals between actual and predicted values.

        Parameters
        ----------
        t_data : np.ndarray
            Time points. Shape: (n_samples,)
        x_data : np.ndarray
            Actual state values. Shape: (n_samples, n_states) or (n_samples,)

        Returns
        -------
        np.ndarray
            Residuals (x_data - predictions). Same shape as x_data.

        Raises
        ------
        RuntimeError
            If the solver has not been fitted.
        ValueError
            If input arrays are invalid.
        """
        if not self._is_fitted:
            raise RuntimeError("Solver must be fitted before computing residuals")

        self._validate_inputs(t_data, x_data, name="compute_residuals")

        predictions = self.predict(t_data)
        return x_data - predictions
