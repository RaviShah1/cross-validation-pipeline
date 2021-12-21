
class Metric:
    def __init__(self, name: str):
        """
        Generates the Metric 

        Parameters
        ---------
        name : str
            the name of the metric
        """
        pass
    
    def __call__(self) -> float:
        """
        Gets the evaluation score of the y_true and y_pred values

        Parameters
        ----------
        y_true : np.ndarray
            the actual values
        y_pred : np.ndarray
            the predicted values
        """
        pass

    def _compute(self) -> float:
        """
        Computes the evalution score

        Parameters
        ----------
        y_true : np.ndarray
            the actual values
        y_pred : np.ndarray
            the predicted values
        """
        pass