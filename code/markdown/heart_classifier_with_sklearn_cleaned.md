# Creating a Scikit-Learn Classifer

### API required to become an estimator

- Class should provide:
  - a `fit()`
    - Given data (features) and target, fits the model to make predictions based on given data
  - a `predict()`
    - Given data (features), returns array of predictions for each observation based on test data in `fit()`
  - a `constructor` (`__init__()`)
    - Can supply default arguments here and does any initialization required
  
- Can provide other useful methods like:
 - `predict_proba()`
 - `score()`

For more information, go to [scikit-learn's website](https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects)

## Creating our `CustomDecisionTreeClassifier` class

Our custom decision tree classifier, which we implemented [here](https://csc466-team7.github.io/csc466_project/#/example/1), needs to be turned into a class. From there, we will add required sklearn estimator checks so that the `check_estimator()` function passes when given our model. This means sklearn has deemed our estimator correct in terms of having the necessary functionality and basic checks to use it as a classifier.

Let's get started by making a class that extends `BaseEstimator` and `ClassifierMixin`, which both provide some helpful functions.

```python
class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, criterion='entropy',default=None):
        self.criterion = criterion
        self.default = default
        
    # Needed for check_estimator
    def _more_tags(self):
        return {
            "poor_score": True
        }
```

We have only written an `entropy` function previously, but if we wanted to have other ways of determining information gain, we could specify it with the `criterion` attribute, as the `DecisionTreeClassifier` sklearn provides does.

We also provide a `default` value for which the user can give to put as the default prediction to use if we make it to a point where the features given create a circumstance where no branch in the decision tree exists.

The `_more_tags()` method is provided here with a return dictionary of `poor_score = True` to let sklearn know that our decision tree may be a poor predictor and that's ok. This is because our decision tree is not effective for quanititative data that sklearn may pass as tests for our classifier. We can make a classifier that does handle continuous variables, though. We have not done that here.

### Writing fit/predict

So what changes here?

Well, `fit()` will do some of the transformations and checks required and then will call our `tree_creation` function to train our classifier from the given training data and `make_rules` to be ready to make predicitions from.

The `predict()` will also do some transformations and checks and then will call our `make_predictions` function on the given dataset.

## The hard part

Now here might be some hard news.

*We are not guaranteed to be passed a **pandas DataFrame** as the X (training set) variable to fit or predict*.

We will assume that we do get at least a 2D matrix that corresponds to a dataframe, though, and a 1D matrix that corresponds to the targets for each observation. This means we cannot rely on `X.columns` to get the columns out of a dataframe and can really only use `numpy` functions during this OR turn it into a `pandas` dataframe ourselves. For this exercise, we used `numpy` functions.

Luckily, sklearn does provide some functions we can use during `fit()` and `predict()` to help do some checks and shaping for us:

- `fit()`
  - `check_X_y(X, y)` - Given `X` and `y`, returns a numpy array for each instead, even if input is `pandas DataFrame` or `Series`
  - `check_classification_targets(y)` - Makes sure values in given `y` lead to a classification problem
  - `np.atleast_2d(X)` and `np.atleast_1d(y)` - checks and coerces given matrix to dimension specified
- `predict()`
  - `check_is_fitted(self)` - makes sure model has been `fit()`
  - `check_array(X)` - From sklearn: "By default, the input is checked to be a non-empty 2D array containing only finite values"
  
### But what if we used DataFrame `columns` to pick certain columns or generate rules?

All is not lost! One way to go about this is to convert the `numpy` arrays to a DataFrame with named columns of your own. This is the easy lame way :D.


Another way is to just keep track of the current indexes of the columns you still have during tree creation. Thus, the starting indexes to `generate_tree` would be
```python
np.asarray(list(range(self.n_features_in_)))
```
where `self.n_features_in_ = X.shape[1]`. This is because we start with all indexes at first, which is just all the features' indexes. On subsequent calls to `generate_tree`, we can pass indexes that relate to the original indexes with all the features.

Let's say we have a decision tree with features: Ethnicity, Age, and Height. Then we pick Age as our best feature (highest info gain). That means we will pass Ethnicity and Height as a new subtree to make. To start, the indexes would have been `0, 1, and 2`. The subtree would get indexes `0 and 2` then.
  - `0 -> Ethnicity`
  - `1 -> Age`
  - `2 -> Height`
  
Thus the function definition for the `_tree_creation` in your class may look like:
```python
def _tree_creation(self,X,y,related_idxs):
```

and will still return a dictionary **where instead of column names you get indexes**.


Some helpful `numpy` functions for dealing with `numpy` arrays:
- `unique(y)` - returns only unique values. Can also return indexes and counts as well!
- `where(condition)` - Given condition traverses array and returns indexes that satisfy condition. Helpful for subsetting based on condition. Can be used similarly to `df[z == 2]` (`np.where(z == 2)` where `z` is numpy array)
- `delete(y, value(s), axis)` - returns new numpy array with value(s) deleted along that axis. If you have 2D matrix and want to delete a column, can do: `np.delete(X, column, axis=1)`
- `numpy` arrays can be subsetted like `pandas Series` can be. `X[:, col]` selects all observations in the 2D matrix `X` at column `col`.


#### The nice part about all of this is that the `generate_rules` function really doesn't have to change since the given tree is still just a dictionary.

### Printing it out with column names

Since we know only have indexes in our tree, if we received a `DataFrame` with columns during fitting, we may want to also print our tree with columns. Here's a helper function to do that:

```python
    def print_decision_tree(self, with_cols=False):
        try:
            getattr(self, "_tree")
        except AttributeError:
            raise RuntimeError("You must train classifer before printing tree!")
            
        self._print_tree_helper(self._tree, with_cols)
            
    def _print_tree_helper(self, tree, replace_cols):
        mytree = copy.deepcopy(tree)
        def fix_keys(tree):
            if type(tree) != dict:
                if type(tree) == np.int64:
                    return int(tree)
            new_tree = {}
            for key in list(tree.keys()):
                if type(key) == np.int64 or type(key) == np.int32:
                    if replace_cols:
                        new_tree[self.cols_[int(key)]] = tree[key]
                    else:
                        new_tree[int(key)] = tree[key]
                else:
                    new_tree[key] = tree[key]
            for key in new_tree.keys():
                new_tree[key] = fix_keys(new_tree[key])
            return new_tree
        mytree = fix_keys(mytree)
        print(json.dumps(mytree, indent=4, sort_keys=True))
```

But where does `self.cols_` come from to do this? You can add a way to grab the columns in fit:
```python
if hasattr(X, 'columns'):
    self.cols_ = X.columns
else:
    self.cols_ = None
```

then all we have to do is call the print correctly:
```python
model.print_decision_tree(with_cols=True)
```

### Now let's check out if our estimator works with the heart dataset again. We should get the same scores!

Once again we will sklearn's `train_test_split` and `accuracy_score` and `f1_score`


```python
X2_train, X2_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state = 0)

clf = CustomDecisionTreeClassifier(default=1)
y_test = clf.fit(X2_train, t_train).predict(X2_test)

print(f'Accuracy: {accuracy_score(y_test, t_test)}')
print(f'F1 score: {f1_score(y_test, t_test)}')
```

    Accuracy: 0.7912087912087912
    F1 score: 0.7999999999999999


### Let's see the decision tree with the columns and indexes next to them as an example of what happens

First a table of the indexes!


```python
print_index_table(clf)
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
      <th>Column Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sex</th>
      <td>0</td>
    </tr>
    <tr>
      <th>cp</th>
      <td>1</td>
    </tr>
    <tr>
      <th>fbs</th>
      <td>2</td>
    </tr>
    <tr>
      <th>restecg</th>
      <td>3</td>
    </tr>
    <tr>
      <th>exang</th>
      <td>4</td>
    </tr>
    <tr>
      <th>slope</th>
      <td>5</td>
    </tr>
    <tr>
      <th>ca</th>
      <td>6</td>
    </tr>
    <tr>
      <th>thal</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_decision_tree_custom(clf.fit(X2_train, t_train))
```

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


We can see that `thal`, which is columns number 7, is the most important feature according to our deicison tree. If `thal` is 0, then `sex`, column number 0, is our next most important feature for the subtree. The actual tree that gets produced, if just using indexes, will just have a `7` instead of `thal` if the column names are not replaced. This is just a visualization of how the conversion can be done eaisly.

Sweet! If you remember the implementation guide for ID3 classifier, those scores are **exactly the same** as what sklearn's `DecisionTreeClassifier` gives back!

### Alright, final check! Let's see if our custom classifier passes as a classifier according to sklearn!

Run `check_estimator` with an instantiation. **_No output is good output!_**


```python
check_estimator(CustomDecisionTreeClassifier())
```
