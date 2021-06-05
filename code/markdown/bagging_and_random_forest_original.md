```python
from sklearn.datasets import make_regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
```

## Ensemble model with Bagging

The guiding principle behind ensemble models is to leverage a combination of weak learners to create a strong learner. Bagging does this by creating subsets of the training data through resampling, and training an ML model of choice (in our example, we will use the Custom Decision Tree model we made in the previous tutorial) on the substes of training data. This produces numerous models, each slightly different than others. By averaging the prediction of these inidividual learners for a given obervation, we should get more robust results that accounts for variance in the test data than we would get from an individual learner.

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

### Bagging (Classifier) with SK Learn

For this example, we will continue working with the heart diseases dataset.


```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
%load_ext autoreload
%autoreload 2
```

First, let's run our previous tutorial to import our custom decision tree classifier from the previous tutorial, as well as the cleaned feature set and target column.


```python
%run heart_classifier_with_sklearn.ipynb
```

    Accuracy: 0.7912087912087912
    F1 score: 0.7999999999999999
    {
        "thal/7": {
            "0": {
                "sex/0": {
                    "0": 1,
                    "1": 0
                }
            },
            "1": {
                "ca/6": {
                    "0": {
                        "restecg/3": {
                            "0": 1,
                            "1": {
                                "slope/5": {
                                    "0": 0,
                                    "1": 0
                                }
                            }
                        }
                    },
                    "1": 0,
                    "2": 0,
                    "3": 0
                }
            },
            "2": {
                "ca/6": {
                    "0": {
                        "restecg/3": {
                            "0": {
                                "cp/1": {
                                    "0": {
                                        "slope/5": {
                                            "1": 1,
                                            "2": {
                                                "sex/0": {
                                                    "0": 1,
                                                    "1": 1
                                                }
                                            }
                                        }
                                    },
                                    "1": {
                                        "sex/0": {
                                            "0": 1,
                                            "1": {
                                                "slope/5": {
                                                    "1": 0,
                                                    "2": 1
                                                }
                                            }
                                        }
                                    },
                                    "2": 1,
                                    "3": {
                                        "sex/0": {
                                            "0": 1,
                                            "1": {
                                                "exang/4": {
                                                    "0": 0,
                                                    "1": 1
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "1": {
                                "cp/1": {
                                    "0": {
                                        "exang/4": {
                                            "0": 1,
                                            "1": {
                                                "slope/5": {
                                                    "1": 0,
                                                    "2": 1
                                                }
                                            }
                                        }
                                    },
                                    "1": 1,
                                    "2": {
                                        "sex/0": {
                                            "0": 1,
                                            "1": {
                                                "exang/4": {
                                                    "0": {
                                                        "slope/5": {
                                                            "0": 1,
                                                            "1": 0,
                                                            "2": 1
                                                        }
                                                    },
                                                    "1": 1
                                                }
                                            }
                                        }
                                    },
                                    "3": 1
                                }
                            },
                            "2": {
                                "cp/1": {
                                    "0": 0,
                                    "2": 1
                                }
                            }
                        }
                    },
                    "1": {
                        "cp/1": {
                            "0": 0,
                            "1": 1,
                            "2": 1,
                            "3": 0
                        }
                    },
                    "2": {
                        "exang/4": {
                            "0": {
                                "fbs/2": {
                                    "0": {
                                        "sex/0": {
                                            "0": 1,
                                            "1": {
                                                "restecg/3": {
                                                    "0": 1,
                                                    "1": 0
                                                }
                                            }
                                        }
                                    },
                                    "1": 0
                                }
                            },
                            "1": 0
                        }
                    },
                    "3": {
                        "cp/1": {
                            "0": 0,
                            "2": {
                                "fbs/2": {
                                    "0": 0,
                                    "1": 1
                                }
                            }
                        }
                    },
                    "4": 1
                }
            },
            "3": {
                "cp/1": {
                    "0": {
                        "ca/6": {
                            "0": {
                                "slope/5": {
                                    "0": 0,
                                    "1": 0,
                                    "2": {
                                        "restecg/3": {
                                            "0": {
                                                "exang/4": {
                                                    "0": 0,
                                                    "1": 0
                                                }
                                            },
                                            "1": {
                                                "exang/4": {
                                                    "0": 1,
                                                    "1": 0
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "1": {
                                "slope/5": {
                                    "1": {
                                        "restecg/3": {
                                            "0": 0,
                                            "1": {
                                                "exang/4": {
                                                    "0": 0,
                                                    "1": 0
                                                }
                                            },
                                            "2": 0
                                        }
                                    },
                                    "2": 0
                                }
                            },
                            "2": 0,
                            "3": 0,
                            "4": 0
                        }
                    },
                    "1": {
                        "slope/5": {
                            "0": 0,
                            "1": 1,
                            "2": {
                                "fbs/2": {
                                    "0": {
                                        "restecg/3": {
                                            "0": 1,
                                            "1": 0
                                        }
                                    },
                                    "1": 1
                                }
                            }
                        }
                    },
                    "2": {
                        "slope/5": {
                            "1": {
                                "ca/6": {
                                    "0": {
                                        "exang/4": {
                                            "0": 1,
                                            "1": {
                                                "fbs/2": {
                                                    "0": {
                                                        "restecg/3": {
                                                            "0": 1,
                                                            "1": 0
                                                        }
                                                    },
                                                    "1": 0
                                                }
                                            }
                                        }
                                    },
                                    "1": 0,
                                    "3": 0
                                }
                            },
                            "2": 1
                        }
                    },
                    "3": {
                        "fbs/2": {
                            "0": {
                                "restecg/3": {
                                    "0": 1,
                                    "1": {
                                        "slope/5": {
                                            "1": 0,
                                            "2": 1
                                        }
                                    }
                                }
                            },
                            "1": 1
                        }
                    }
                }
            }
        }
    }



```python
X = X2 # From the previous notebook
y = t # From the previous notebook
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
      <th>sex</th>
      <th>cp</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>exang</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



    0      1
    1      1
    2      1
    3      1
    4      1
          ..
    298    0
    299    0
    300    0
    301    0
    302    0
    Name: disease_present, Length: 303, dtype: int64


We will now evaluate the `f1` score for an ensemble model with the following number of learners: 1, 10, 25, 50

<qinline>

<question>

Before we proceed with that, what would you expect to see happen to the `f1` score as the number of trees increases?

</question>

<answer>

It should get better since we have more learners and thus more variance to make a prediction.

</answer>

</qinline>


```python
scores = {1: [], 10: [], 25: [], 50: []}
for num_estimators in scores:
  # define the model
  model = BaggingClassifier(n_estimators=num_estimators, base_estimator=CustomDecisionTreeClassifier())
  # evaluate the model
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
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
      <td>0.774194</td>
      <td>0.750000</td>
      <td>0.787879</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.687500</td>
      <td>0.687500</td>
      <td>0.727273</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.687500</td>
      <td>0.857143</td>
      <td>0.888889</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.583333</td>
      <td>0.454545</td>
      <td>0.560000</td>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.787879</td>
      <td>0.750000</td>
      <td>0.764706</td>
      <td>0.727273</td>
    </tr>
  </tbody>
</table>
</div>


Now that we have multiple scores for various values of `n_estimators`, let's see how the number of estimators fare against each other by averaging the scores for each respective value of `n_estimators`. 

Here, `n_estimators` is the number of trees in the ensemble model, and we are interested in how this makes a difference to support our understanding of the benefits of bagging to create strong learners.


```python
scores_df.mean()
```




    1     0.756065
    10    0.778015
    25    0.787050
    50    0.778031
    dtype: float64



It is clear from the above, that as the number of estimators (i.e trees) increases, the ensemble model produces more robust predictions in general. Since our decision tree classifier is a really strong learner for the data set we have, there is not much variation.

## Extending bagging with Random Forests

Random forests is very similar algorithm to bagging, with the addition of dropping a few features in the training data (i.e only using a subset of features instead of all of them, chose randomly) for each iteration along with resampling it. This adds another level of randmness to the generationg of trees, and further account for variance. 

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

### Random Forest (Classifier) with SK Learn


```python
scores = {1: [], 10: [], 25: [], 50: []}
for num_estimators in scores:
  # define the model
  model = RandomForestClassifier(n_estimators=num_estimators)
  # evaluate the model
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
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
      <td>0.733333</td>
      <td>0.774194</td>
      <td>0.812500</td>
      <td>0.812500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.687500</td>
      <td>0.800000</td>
      <td>0.787879</td>
      <td>0.787879</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.857143</td>
      <td>0.914286</td>
      <td>0.888889</td>
      <td>0.918919</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.620690</td>
      <td>0.615385</td>
      <td>0.640000</td>
      <td>0.615385</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.727273</td>
      <td>0.823529</td>
      <td>0.857143</td>
      <td>0.823529</td>
    </tr>
  </tbody>
</table>
</div>


Let's perform the same analysis for the number of estimator that we did above. 


```python
scores_df.mean()
```




    1     0.787345
    10    0.819488
    25    0.824294
    50    0.818179
    dtype: float64



The results are consistent with what we observed above. Increasing the number of estimators improves the performance of the ensemble model, supporting the benefit of using an ensemble model as opposed to a single learner. 
