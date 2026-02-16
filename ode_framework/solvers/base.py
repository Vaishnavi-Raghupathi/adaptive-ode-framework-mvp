"""Base classes for ODE solvers."""
from abc import ABC, abstractmethod

import numpy as np


class BaseSolver(ABC):
    """Abstract base class for ODE solvers.

    This class defines the interface that all ODE solver implementations must follow.
    Subclasses should implement the abstract methods to provide specific solver
    functionality.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def fit(self, t_data: np.ndarray, x_data: np.ndarray) -> None:
        """Fit the solver to training data.

        This method should train or configure the solver based on the provided
        time series data and corresponding state values.

        Parameters
        ----------
        t_data : np.ndarray
            Time points of the training data. Shape: (n_samples,)
        x_data : np.ndarray
            State values at corresponding time points. Shape: (n_samples, n_states)
            or (n_samples,) for univariate systems.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the input arrays have incompatible shapes or invalid data.
        """

    @abstractmethod
    def predict(self, t_eval: np.ndarray) -> np.ndarray:
        """Predict state values at given time points.

        This method uses the fitted model to predict system states at new
        time points.

        Parameters
        ----------
        t_eval : np.ndarray
            Time points at which to evaluate the solution. Shape: (n_eval,)

        Returns
        -------
        np.ndarray
            Predicted state values at evaluation times. Shape: (n_eval, n_states)
            or (n_eval,) for univariate systems.

        Raises
        ------
        RuntimeError
            If the solver has not been fitted before prediction.
        """

    @abstractmethod
    def compute_residuals(self, t_data: np.ndarray, x_data: np.ndarray) -> np.ndarray:
        """Compute residuals between predicted and actual values.

        This method calculates the residuals (differences) between the solver's
        predictions and the actual data at the given time points.

        Parameters
        ----------
        t_data : np.ndarray
            Time points where residuals should be computed. Shape: (n_samples,)
        x_data : np.ndarray
            Actual state values at the time points. Shape: (n_samples, n_states)
            or (n_samples,) for univariate systems.

        Returns
        -------
        np.ndarray
            Residual values (x_data - predictions). Shape: same as x_data.

        Raises
        ------
        RuntimeError
            If the solver has not been fitted before computing residuals.
        """
