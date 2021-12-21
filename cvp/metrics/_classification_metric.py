from sklearn import metrics as skmetrics
import numpy as np
from ._metric import Metric

class ClassificationMetric(Metric):
    """
    Classification Metric 
    Computes an evalution of data based on a specific classification metric
    """

    def __init__(self, name: str):
        """
        Generates the ClassificationMetric 

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
        if self.name == 'accuracy':
            return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        elif self.name == 'f1':
            return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
        elif self.name == 'precision':
            return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
        elif self.name == 'recall':
            return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
        elif self.name == 'auc':
            return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        elif self.name == 'logloss':
            return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)
        else:
            raise Exception("Not Implemented")
