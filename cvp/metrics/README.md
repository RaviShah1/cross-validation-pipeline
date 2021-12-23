# Metrics Documentation

This is a wrapper for the sklearn metrics module.

## Classification Options

```python
from cvp.metrics import ClassificationMetric
metric = ClassificationMetric('accuracy') # the parameter can be any option from the table below
metric(y_true, y_pred)
```

<table>
  <tr>
    <td>accuracy</td>
    <td>returns the accuracy score</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>returns the f1 score</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>returns the precision score</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>returns the recall score</td>
  </tr>
  <tr>
    <td>auc</td>
    <td>returns the roc auc score</td>
  </tr>
  <tr>
    <td>auc_multi</td>
    <td>returns the roc auc score for multiclass classification</td>
  </tr>
  <tr>
    <td>logloss</td>
    <td>returns the log loss score</td>
  </tr>
  </table>

## Regression Options

```python
from cvp.metrics import RegressionMetric
metric = RegressionMetric('rmse') # the parameter can be any option from the table below
metric(y_true, y_pred)
```

<table>
  <tr>
    <td>mae</td>
    <td>returns the mean absolute error</td>
  </tr>
  <tr>
    <td>mse</td>
    <td>returns the mean squared</td>
  </tr>
  <tr>
    <td>rmse</td>
    <td>returns the root mean squared error</td>
  </tr>
  <tr>
    <td>msle</td>
    <td>returns the mean squared log loss</td>
  </tr>
  <tr>
    <td>rmsle</td>
    <td>returns the root mean squared log loss</td>
  </tr>
  <tr>
    <td>r2</td>
    <td>returns the r2 score</td>
  </tr>
  </table>

## Adding a new Metric

Using a metric thats not in the API? Use the Metric interface!

```python
from cvp.metrics import Metric

class MyMetric(Metric):
    def __init__(self, name: str):
        self.name = name

    def __call__(self):
        return self._compute()

    def _compute(self):
        return # write your metric here
```
