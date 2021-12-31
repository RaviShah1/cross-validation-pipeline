# Splits Documentation

This contains a class to split your data into folds for cross validation.

## How to Use

```python
from cvp.splits import Split
splitter = Split(data, ['f1', 'f2', 'f3'], 'label')
splitter.split()
```

Split Parameters:

<table>
  <tr>
    <td>data</td>
    <td>pd.DataFrame</td>
    <td>the dataset to split</td>
  </tr>
  <tr>
    <td>X</td>
    <td>list</td>
    <td>the column names of the input features</td>
  </tr>
  <tr>
    <td>y</td>
    <td>str</td>
    <td>the column name of the output</td>
  </tr>
  <tr>
    <td>n_splits</td>
    <td>int</td>
    <td>the number of folds to create</td>
  </tr>
  <tr>
    <td>method</td>
    <td>str</td>
    <td>how to split the data</td>
  </tr>
  <tr>
    <td>groups</td>
    <td>str</td>
    <td>the group column for group-kfold</td>
  </tr>
  <tr>
    <td>holdout</td>
    <td>bool</td>
    <td>true if you want a holdout instead of folds</td>
  </tr>
  <tr>
    <td>holdout_fold</td>
    <td>int</td>
    <td>a fold to holdout if holdout is true</td>
  </tr>
  <tr>
    <td>shuffle</td>
    <td>bool</td>
    <td>true if you want to shuffle the dataframe</td>
  </tr>
  <tr>
    <td>seed</td>
    <td>int</td>
    <td>the random state seed</td>
  </tr>
  </table>

## K-Fold

```python
splitter = Split(data, ['f1', 'f2', 'f3'], 'label', 5)
splitter.split()
```

## Stratified K-Fold

```python
splitter = Split(data, ['f1', 'f2', 'f3'], 'label', 5, 'skf')
splitter.split()
```

## Group K-Fold

```python
splitter = Split(data, ['f1', 'f2', 'f3'], 'label', 5, 'gkf', 'group')
splitter.split()
```

## Holdout

```python
splitter = Split(data, ['f1', 'f2', 'f3'], 'label', 5, holdout=True, holdout_fold=0)
splitter.split()
```
