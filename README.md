# cross-validation-pipeline

Cross Validation Pipeline (cvp) is a future open source python package. CVP is a fast and simple way to implement a cross validation framework and pipeline to a machine learning project

## Overview
<table>
  <tr>
    <td>cvp</td>
    <td>Simple and fast cross validation pipeline</td>
  </tr>
  <tr>
    <td>cvp.metrics</td>
    <td>Evalution Metrics for Classification and Regression models</td>
  </tr>
  <tr>
    <td>cvp.splits</td>
    <td>Splits a dataset using any type of cross validation framework (e.g. kfold, holdout, etc.)</td>
  </tr>
  <tr>
    <td>cvp.model_pipeline</td>
    <td>Applies a full cross validation pipeline to a fit_predict type machine learning model</td>
  </tr>
</table>

## Installation

Currently Unavailable

## Getting Started

```python
from cvp.splits import Split
from cvp.metrics import RegressionMetric
from cvp.model_pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd

data = pd.read_csv("../input/train.csv")
X = ['feature1', 'feature2', 'feature3']
y = 'label'

splitter = Split(data, X, y, 5, 'skf')
metric = RegressionMetric('rmse')
model = XGBClassifier()

pipeline = Pipeline(data, X, y, model, splitter, metric, False, 'xgb', '/saved_models')
results = pipeline.run()
```

## Resources
- [Documentation](https://github.com/RaviShah1/cross-validation-pipeline/blob/main/README.md)

