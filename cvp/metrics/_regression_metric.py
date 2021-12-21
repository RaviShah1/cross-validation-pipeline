from sklearn import metrics as skmetrics
import numpy as np
from ._metric import Metric

class RegressionMetric(Metric):
    """
    Regression Metric 
    Computes an evalution of data based on a specific regression metric
    """

    def __init__(self, name: str):
        """
        Generates the RegressionMetric 

        Parameters
        ---------
        name : str
            the name of the metric
        """
        self.name = name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Gets the evaluation score of the y_true and y_pred values

        Parameters
        ----------
        y_true : np.ndarray
            the actual values
        y_pred : np.ndarray
            the predicted values
        """
        return self._compute(y_true, y_pred)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the evalution score

        Parameters
        ----------
        y_true : np.ndarray
            the actual values
        y_pred : np.ndarray
            the predicted values
        """
        if self.name == 'mae':
            return skmetrics.mean_absolute_error(y_true, y_pred)
        elif self.name == 'mse':
            return skmetrics.mean_squared_error(y_true, y_pred)
        elif self.name == 'rmse':
            return np.sqrt(skmetrics.mean_squared_error(y_true, y_pred))
        elif self.name == 'msle':
            return skmetrics.mean_squared_log_error(y_true, y_pred)
        elif self.name == 'rmsle':
            return np.sqrt(skmetrics.mean_squared_log_error(y_true, y_pred))
        elif self.name == 'r2':
            return skmetrics.r2_score(y_true, y_pred)
        else:
            raise Exception("Not Implemented")
