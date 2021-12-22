# Metrics Documentation

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

