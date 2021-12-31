import numpy as np
import pandas as pd
from scipy import stats

class Predictor:
    """ 
    Makes a prediction using the output from a Pipeline
    """
    def __init__(self, results: dict, merge: str='mean', holdout: bool=False):
        """
        Creates a Predictor validation pipeline

        Parameters
        ----------
        results : dict
            the output results from a run() of a Pipeline
        merge : str
            how to merge the results from each fold
        holdout : bool
            true if the model was trained with a holdout split
        """
        self.results = results
        self.merge = merge
        self.holdout = holdout

    def predict(self, data: pd.DataFrame, X: list):
        """
        Makes a prediction

        Parameters
        ----------
        data : pd.DataFrame
            the dataset to predict
        X : list
            a list of column names of the input features
        """
        return self._predict(data[X])

    def predict_one(self, X: list):
        """
        Makes a prediction for a single sample

        Parameters
        ----------
        X : list
            the input
        """
        return self._predict(X)

    def _predict(self, inputs):
        """
        Makes the prediction
        """

        if self.holdout:
            model = self.results['model']
            outputs = model.predict(inputs)
            return outputs

        models = self.results['models']
        outputs = list()
        for model in models:
            outputs.append(model.predict(inputs))

        outputs = np.vstack(outputs)
        if self.merge == 'mode':
            return stats.mode(outputs).mode
        elif self.merge == 'mean':
            return np.mean(outputs, axis=0)
        else:
            raise Exception("Not Implemented")