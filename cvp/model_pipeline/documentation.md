# Model Pipeline Documentation

## Example
```python
from cvp.splits import Split
from cvp.metrics import ClassificationMetric
from cvp.model_pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd
from sklearn import datasets

# read in data
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=['f1', 'f2', 'f3', 'f4'])
data['label'] = iris.target

# set configs
X = ['f1', 'f2', 'f3', 'f4']
y = 'label'
splitter = Split(data, X, y, 3, 'skf')
metric = ClassificationMetric('accuracy')
model = XGBClassifier()

# run pipeline
pipeline = Pipeline(data, X, y, model, splitter, metric, False, 'xgb', '.')
results = pipeline.run()
```
