import pandas as pd
import numpy as np
import pickle
import copy
from ..splits import Split
from ..metrics import Metric

class Pipeline:
    """
    Full cross validation pipeline that evaluates trains and evaluates any fit_predict model
    """
    def __init__(self,
                 data: pd.DataFrame,
                 X: list,
                 y: str,
                 model: object,
                 split: Split,
                 metric: Metric,
                 proba: bool,
                 model_name: str,
                 save_dir: str):
        """
        Creates a cross validation pipeline

        Parameters
        ----------
        data : pd.DataFrame
            the dataset
        X : list
            the input features
        y: str | list
            the output values
        model : object
            a fit_predict machine learning model
        split : Split
            a cross validation split
        metric : Metric
            an evaluation metric
        proba : bool
            whether to use .predict_proba()
        model_name : str
            name of the model
        save_dir : str
            where to save the models
        """
        self.df = data
        self.X = X
        self.y = y
        self.model = model
        self.splitter = split
        self.metric = metric
        self.proba = proba
        self.model_name = model_name
        self.save_dir = save_dir

    def run(self) -> dict:
        """
        Runs the Pipeline
        """
        # Split the Data
        self.df = self.splitter.split()
        self.folds = self.splitter.n
        self.holdout = self.splitter.holdout

        # Check if holdout
        if self.holdout:
            return self._train_holdout()

        return self._train()

    def _train_holdout(self) -> dict:
        # Assign Variables
        train = self.df[self.df.fold == 1]
        test = self.df[self.df.fold == 0]

        X_train = train[self.X]
        X_test = test[self.X]
        y_train = train[self.y]
        y_test = test[self.y]

        # Fit Model
        model = copy.deepcopy(self.model)
        model.fit(X_train, y_train)

        # Caclulate Out Of Sample Predictions
        y_pred = None
        if self.proba:
            y_pred = model.predict_proba()
        else:
            y_pred = model.predict(X_test)
        score = self.metric(y_test, y_pred)

        # Save
        pickle.dump(model, open(self.save_dir, 'wb'))

        return {
            'model' : model,
            'save'  : self.save_dir,
            'score' : score,
            'oos'   : y_pred
        }

    def _train(self) -> dict:
        """
        Train models
        """
        models = list()
        saves = list()
        scores = list()
        y_predictions = list()

        for f in range(self.folds):
            # Assign Variables
            train = self.df[self.df.fold != f]
            test = self.df[self.df.fold == f]

            X_train = train[self.X]
            X_test = test[self.X]
            y_train = train[self.y]
            y_test = test[self.y]

            # Fit Model
            model = copy.deepcopy(self.model)
            model.fit(X_train, y_train)

            # Caclulate Out Of Sample Predictions
            y_pred = None
            if self.proba:
                y_pred = model.predict_proba()
            else:
                y_pred = model.predict(X_test)
            score = self.metric(y_test, y_pred)

            # Save
            models.append(model)
            saves.append(f"{self.save_dir}/{self.model_name}_{f}.pkl")
            pickle.dump(model, open(f"{self.save_dir}/{self.model_name}_{f}.pkl", 'wb'))

            # Store results
            scores.append(score)
            y_predictions += list(y_pred)

        y_valid = list()
        for f in range(self.folds):
            y_test = self.df[self.df.fold==f]
            y_valid += list(y_test[self.y].values)
        y_valid
        oof = pd.DataFrame()
        oof['y_true'] = y_valid
        oof[f'{self.model_name}_preds'] = y_predictions

        return {
            'models'    : np.array(models),
            'saves'     : np.array(saves),
            'avg_score' : np.mean(np.array(scores)),
            'scores'    : np.array(scores),
            'oof'       : oof
        }