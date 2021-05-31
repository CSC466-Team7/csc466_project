## Ensemble model with Bagging

The guiding principle behind ensemble models is to leverage a combination of weak learners to create a strong learner. Bagging does this by creating subsets of the training data through resampling, and training an ML model of choice (in our example, we will use Decision Trees) on the substes of training data. This produces numerous models, each slightly different than others. By averaging the prediction of these inidividual learners for a given obervation, we should get more robust results that accounts for variance in the test data than we would get from an individual learner.

### Pseudocode for Bagging


```python
"""
Create our strong learner by bagging weak learners. Note that the code below 
will not run, it is only an outline of the general implementation.
"""
# Assume `data` is defined
trees = []
number_of_trees = 100
for i in range(number_of_trees):
  subset_data = resample(data)
  tree = DecisionTreeModel().fit(subset_data)
  trees.append(tree)

"""
Predict for target variable by running all weak learners on an observation
and averaging the result (or taking the mode if target variable is categorical).
"""
# Assume `x_test` is defined where x_test is the observation we will to predict for
results = []
for tree in trees:
  tree.predict(x_test)
pred = results.mean()
```

### Bagging (Regressor) with SK Learn


```python
from sklearn.datasets import make_regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
```

For this example, we will load the boston home prices dataset provided by sklearn. The target variable will be the median home prices.

Learn more about this dataset [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset).


```python
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
display(X.head())
display(y)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>



    0      24.0
    1      21.6
    2      34.7
    3      33.4
    4      36.2
           ... 
    501    22.4
    502    20.6
    503    23.9
    504    22.0
    505    11.9
    Length: 506, dtype: float64


We will now evaluate the `mean_absolute_error` for an ensemble model with the following number of learners: 1, 10, 25, 50


```python
scores = {1: [], 10: [], 25: [], 50: []}
for num_estimators in scores:
  # define the model
  model = BaggingRegressor(n_estimators=num_estimators)
  # evaluate the model
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
  scores[num_estimators] = n_scores
# report performance
scores_df = pd.DataFrame.from_dict(scores)
display(scores_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>10</th>
      <th>25</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-3.303922</td>
      <td>-2.033333</td>
      <td>-2.083608</td>
      <td>-1.917373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.456863</td>
      <td>-2.463529</td>
      <td>-2.335451</td>
      <td>-2.376196</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.947059</td>
      <td>-1.907647</td>
      <td>-1.956471</td>
      <td>-1.782588</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4.343137</td>
      <td>-3.540588</td>
      <td>-3.182196</td>
      <td>-3.291843</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.270588</td>
      <td>-2.021765</td>
      <td>-1.880706</td>
      <td>-1.838314</td>
    </tr>
  </tbody>
</table>
</div>


Now that we have multiple scores for various values of `n_estimators`, let's see how the number of estimators fare against each other by averaging the scores for each respective value of `n_estimators`. 

Here, `n_estimators` is the number of trees in the ensemble model, and we are interested in how this makes a difference to support our understanding of the benefits of bagging to create strong learners.


```python
scores_df.mean()
```




    1    -3.227508
    10   -2.334098
    25   -2.231449
    50   -2.169995
    dtype: float64



It is clear from the above, that as the number of estimators (i.e trees) increases, the ensemble model produces more robust predictions.

## Extending bagging with Random Forests

Random forests is very similar to bagging, with the addition of dropping a few features in the training data (i.e only using a subset of features instead of all of them, chose randomly) for each iteration along with resampling it. This adds another level of randmness to the generationg of trees, and further account for variance. 

### Pseudocode for Random Forests
The pseudocode below is **very** similar to the one above, except for the `resample` line.


```python
"""
Create our strong learner by bagging weak learners. Note that the code below 
will not run, it is only an outline of the general implementation.
"""
# Assume `data` is defined
trees = []
number_of_trees = 100
for i in range(number_of_trees):
  subset_data = drop_random_features(resample(data))
  tree = DecisionTreeModel().fit(subset_data)
  trees.append(tree)

"""
Predict for target variable by running all weak learners on an observation
and averaging the result (or taking the mode if target variable is categorical).
"""
# Assume `x_test` is defined where x_test is the observation we will to predict for
results = []
for tree in trees:
  tree.predict(x_test)
pred = results.mean()
```

### Random Forest (Regressor) with SK Learn


```python
scores = {1: [], 10: [], 25: [], 50: []}
for num_estimators in scores:
  # define the model
  model = RandomForestRegressor(n_estimators=num_estimators)
  # evaluate the model
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
  scores[num_estimators] = n_scores
# report performance
scores_df = pd.DataFrame.from_dict(scores)
display(scores_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>10</th>
      <th>25</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-3.611765</td>
      <td>-2.184118</td>
      <td>-1.848000</td>
      <td>-1.870196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.288235</td>
      <td>-2.633725</td>
      <td>-2.332784</td>
      <td>-2.465294</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.001961</td>
      <td>-1.844706</td>
      <td>-1.761569</td>
      <td>-1.761176</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.907843</td>
      <td>-3.690588</td>
      <td>-3.386431</td>
      <td>-3.418706</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.827451</td>
      <td>-1.878235</td>
      <td>-1.827216</td>
      <td>-1.787961</td>
    </tr>
  </tbody>
</table>
</div>


Let's perform the same analysis for the number of estimator that we did above. 


```python
scores_df.mean()
```




    1    -3.196889
    10   -2.314654
    25   -2.205537
    50   -2.178932
    dtype: float64



The results are consistent with what we observed above. Increasing the number of estimators improves the performance of the ensemble model, supporting the benefit of using an ensemble model as opposed to a single learner. 
