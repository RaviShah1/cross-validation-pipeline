import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

class Split:
    """
    Splits a dataset acording to a given cross validation framework
    """
    def __init__(self,
                 data: pd.DataFrame,
                 X: list,
                 y: str,
                 n_splits: int = 5,
                 method: str='kf',
                 groups: str=None,
                 holdout: bool=False,
                 holdout_fold: int=0,
                 shuffle: bool=True,
                 seed: int=42):
        """
        Defines the parameters for the split

        Parameters
        ----------
        data : pd.DataFrame
            the dataset
        X : list
            the train features
        y : str | list
            the true prediction values
        n_splits : int
            the number of splits
        method : str
            how to split the data
        groups : str
            the group column for group kfold
        holdout : bool
            whether or not the data is a holdout set
        holdout_fold : int
            the fold to holdout (if any)
        shuffle: bool
            whether or not to shuffle the data
        seed : int
            the random state of the split
        """
        self.df = data
        self.X = data[X]
        self.y = data[y]
        self.n = n_splits
        self.method = method
        if groups is not None:
            self.groups = data[groups]
        self.holdout = holdout
        self.holdout_fold = holdout_fold
        self.shuffle = shuffle
        self.seed = seed
        
    def split(self) -> pd.DataFrame:
        """
        Splits the data

        Returns
        -------
        A dataframe with a fold column
        """
        if self.method == 'kf':
            self._setup_kfold()
        elif self.method == 'skf':
            self._setup_skfold()
        elif self.method == 'gkf':
            self._setup_gkfold()
        else:
            raise Exception("Not Implemented")

        if self.holdout:
            self._setup_holdout()

        return self.df

    def _setup_kfold(self):
        """ Sets up a kfold split """
        kf = KFold(n_splits=self.n, shuffle=self.shuffle, random_state=self.seed)
        for f, (t_, v_) in enumerate(kf.split(self.X)):
            self.df.loc[v_, 'fold'] = f

    def _setup_skfold(self):
        """ Sets up a stratified kfold split """
        skf = StratifiedKFold(n_splits=self.n, shuffle=self.shuffle, random_state=self.seed)
        for f, (t_, v_) in enumerate(skf.split(self.X, self.y)):
            self.df.loc[v_, 'fold'] = f

    def _setup_gkfold(self):
        """ Sets up a group kfold split """
        gkf = GroupKFold(n_splits=self.n)
        for f, (t_, v_) in enumerate(gkf.split(self.X, self.y, self.groups)):
            self.df.loc[v_, 'fold'] = f

    def _setup_holdout(self):
        """ Sets up a holdout split """
        self.df['fold'] = (self.df['fold'] != self.holdout_fold).astype(int)