# Predicting Heart Disease

## Setting up Decision Tree (including continuous features)

### We will be implementing a decision tree that uses entropy to determine information gain for deciding when to split.

If confused you're confused about what entropy and information gain is, check out the [introduction on decision trees](https://csc466-team7.github.io/csc466_project/#/introduction)
 
The implementation details with explanations for entropy and information gain can be found in in the [Heart Decision Tree Classifier notebook](https://csc466-team7.github.io/csc466_project/#/heart_decision_tree_classifier)

### Selecting the Best Split

Now that we have a way to determine the information gain of any given feature, we can go ahead and select the best feature at an arbitrary point in our decision tree. To do this, we just need to loop over all features and select the one with the highest gain.

Continuous features (such as age) pose a challenge to us though, since we have to do more than just a simple equality comparison when checking entropy. When we encounter a continuous variable, we will find go through all possible splits to find the location that gives the maximal information gain. After we find this split location, we can then (effectively) bin this feature and treat it as a categorical variable while building our tree.

Finding the optimal split for continuous features must be computed each time we want to make a new split, as previous splits have likely changed the values remaining in this column. This means the optimal split of any feature is likely to be different throughout the tree.

<qinline>

<question>

Why do we have to find a split and bin continuous features when building our tree?

</question>

<answer>

If we leave these features as continuous, there will be minimal information gain if this feature is chosen while building our tree. For example, consider the following.
    
We have `ages=[20, 30, 40, 40, 50]` and `targets=[0, 1, 0, 1, 0]` respectively. Without computing actual information gain and for the sake of example, we can see that choosing `age=30` as our split point means that we are left with two paths. One with `age=30` (with only one person matching this criteria) and one with `age!=30` (of which, 4 people live).
    
This is clearly not optimal as this process could be repeated for the remaining people in the `age!=30` category, leading to an overfitted tree. Instead, we find the use our methd of finding the split location that gives the maximal gain and bin our variables according to this.
    
</answer>

</qinline>


```python
# Finds the optimal feature and information gain of dataset X
def select_split(X, y):
    col = None
    gr = 0
    
    for c in X.columns:
        cur_gain_ratio = 0
        
        # if we encounter a continuous variable, find the optimal split
        if X[c].dtypes == "int64" or X[c].dtypes == "float64":
            c, cur_gain_ratio = cont_split(X, y, c)
        else:
            cur_gain_ratio = gain(y, X[c])
        
        if cur_gain_ratio > gr:
            gr = cur_gain_ratio
            col = c
    
    return col, gr


def cont_split(X, y, col):
    # order the variables in the series
    xs = sorted(X[col].unique().tolist())
    
    best_split = None
    gr = 0
    
    for i in range(len(xs) - 1):
        split = round((xs[i + 1] + xs[i]) / 2, 2)
        
        # creates a binned series
        s = X[col] < split
        split_gain = gain(y, s)
                
        if split_gain >= gr:
            gr = split_gain
            best_split = split
    
    best_split = f'{best_split:.2f}'
    col_name = f"{col}<{best_split}"
    return col_name, gr
```

## Creating our Tree

Actually creating the tree at this point follows a very similiar process to what we did in the [Heart Decision Tree Classifier notebook](https://csc466-team7.github.io/csc466_project/#/heart_decision_tree_classifier). Now, we just have to handle the true and false conditions returned by our continuous splits. Before we do this though, we will be **adding** a base case!

### Base Cases

1. There are no more features, so we select the most common target class.

2. The number of unique classes left in the target is 1. *This means that the entropy of target is 0* so we know what selection to make... the only value in the target left.

3. *New:* The number of items is less than our min_split_count. So we'll return the most common target class again.

<qinline>

<question>

Why might we want to stop tree creation when only a certain amount of items remain?
    
</question>

<answer>

If we continue making splits when `n < min_split_count`, we run into the risk of overfitting our tree (i.e. our tree becomes too specific for the training set we provide).

</answer>

</qinline>

### Recursive Part

At this point, if we haven't hit a base case, we know we've found a valid split for our tree. This means we know want to split our tree into each class of the feature we found. After that, we need to make a new tree with the features left after the selection of this given feature. This is nearly identical to the process found in the [Heart Decision Tree Classifier notebook](https://csc466-team7.github.io/csc466_project/#/heart_decision_tree_classifier), but we must now be careful with the continous splits on continuous variables since we are checking relationships between values, not equality.  For completeness, the steps are:

- Go through each unique class, `c`, in the feature, `B`
  - Create a new observation list where the observations (targets and other features) have `B = c` in that observation
      ```
      # for categorical variables
      indexes = where(B = c)
      
      # for continuous variables
      indexes = where(B < c)
      indexes = where(B >= c)
      
      new_X = X.locate_by_index[indexes]
      new_y = y.locate_by_index[indexes]
      ```
  - Remove feature `B` from our observations
      ```
      new_X.drop(columns=[col])
      ```
  - Add to the tree at feature `B` class `c` whose value is a new tree
     ```
     tree[B][c] = tree(new_X, new_y)
     ```

### Generating the Rules

The way we generate a ruleset does not change just because we have continuous variables. We create them by using the tree produced. If you're interested in seeing how we go about generating it, check out the [Heart Decision Tree Classifier notebook](https://csc466-team7.github.io/csc466_project/#/heart_decision_tree_classifier).

Note: Generating a ruleset is not required. It is merely a nice way to visualize the paths created by the tree.

## Making Predictions

Finally we are near the end! Now that we have our tree or rules, we can start to make predictions.

Let's again refer to the [introduction on decision trees](https://csc466-team7.github.io/csc466_project/#/introduction). Let's say our observation is `x` and contains each feature we used to make our tree. Since `Weather` is our root feature, we need to find what `Weather` in `x`. If `x.Weather` is `warm`, we look at `x.Day of the Week` next. If that is `weekend`, we should pick `Hawaiian shirt`.

But what if we get to `x.Weather` and it's `hot`? Then we have no information in our tree of what to do!!! Thus, we will just make our best guess. To do that, we can take a default value that the user provides *or* we can look at the target values up to that point and pick the class in target that occurs the most often. Otherwise, we try and recurse on the sub-tree we get when choosing the related class in the tree. If there is no classes left and we are at a prediction in the tree instead, we just return that prediction.

```
Input: x - observation
       tree - current tree
       default - what to return if nothing left in tree

if tree.is_leaf():
    return tree.value

feature_to_use = tree.root
observation_class = x[feature_to_use]

if observation_class not in tree:
    return default

return make_prediction(tree.go_to(feature_to_use), x, default)
```

If you made a dictionary for your tree, you can use the `print_tree` function below to see what your function generated.


```python
# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))
```

## Testing your Tree

- We will use `sklearn`'s train_test_split to see how we did

You can find the dataset and download it from the website. After that, just set the appropriate variables!
```
heart_disease = pd.read_csv("path to heart_disease.csv")
X2 = heart_disease.drop(columns=["disease_present"])
t = heart_disease["disease_present"]
```


```python
default = 0
from sklearn.model_selection import train_test_split

X2_train, X2_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state = 0)

tree = tree_creation(X2_train,t_train)
rules = generate_rules(tree)

y_test = X2_test.apply(lambda x: make_prediction(rules,x,default),axis=1)
```

### What does our tree look like?


```python
print_tree(tree)
```

    {
        "thal": {
            "0": 0,
            "1": {
                "ca": {
                    "0": {
                        "age<48.00": {
                            "False": 1,
                            "True": 0
                        }
                    },
                    "1": 0,
                    "2": 0,
                    "3": 0
                }
            },
            "2": {
                "ca": {
                    "0": {
                        "chol<228.50": {
                            "False": {
                                "oldpeak<1.65": {
                                    "False": 0,
                                    "True": {
                                        "age<58.50": {
                                            "False": {
                                                "thalach<170.00": {
                                                    "False": 1,
                                                    "True": {
                                                        "cp": {
                                                            "0": {
                                                                "sex": {
                                                                    "0": 0,
                                                                    "1": 1
                                                                }
                                                            },
                                                            "2": 1,
                                                            "3": 0
                                                        }
                                                    }
                                                }
                                            },
                                            "True": {
                                                "trestbps<109.00": {
                                                    "False": 1,
                                                    "True": 1
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "True": 1
                        }
                    },
                    "1": {
                        "cp": {
                            "0": 0,
                            "1": 1,
                            "2": 1,
                            "3": 0
                        }
                    },
                    "2": {
                        "age<63.50": {
                            "False": 1,
                            "True": {
                                "trestbps<138.00": {
                                    "False": 1,
                                    "True": 0
                                }
                            }
                        }
                    },
                    "3": {
                        "thalach<149.50": {
                            "False": 1,
                            "True": 0
                        }
                    },
                    "4": 1
                }
            },
            "3": {
                "cp": {
                    "0": {
                        "oldpeak<0.60": {
                            "False": 0,
                            "True": {
                                "thalach<117.00": {
                                    "False": {
                                        "chol<217.00": {
                                            "False": 0,
                                            "True": {
                                                "trestbps<136.00": {
                                                    "False": 0,
                                                    "True": 1
                                                }
                                            }
                                        }
                                    },
                                    "True": 1
                                }
                            }
                        }
                    },
                    "1": {
                        "chol<225.00": {
                            "False": 0,
                            "True": 1
                        }
                    },
                    "2": {
                        "oldpeak<1.70": {
                            "False": 0,
                            "True": {
                                "trestbps<124.00": {
                                    "False": 0,
                                    "True": 1
                                }
                            }
                        }
                    },
                    "3": {
                        "age<48.50": {
                            "False": 1,
                            "True": 0
                        }
                    }
                }
            }
        }
    }


### How did our decision tree do with the test set?


```python
from sklearn.metrics import accuracy_score, f1_score

print(f'Accuracy: {accuracy_score(y_test, t_test)}')
print(f'F1 score: {f1_score(y_test, t_test)}')
```

    Accuracy: 0.7582417582417582
    F1 score: 0.7555555555555555


### How does this compare with the battle-tried SciKit Learn version?


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=5)
model = clf.fit(X2_train, t_train)

y_model_test = model.predict(X2_test)

print(f'Accuracy: {accuracy_score(y_model_test, t_test)}')
print(f'F1 score: {f1_score(y_model_test, t_test)}')
```

    Accuracy: 0.7912087912087912
    F1 score: 0.8080808080808081


# Not bad!

The difference here is likely due to the 

<qinline>

<question>

How else could we handle continuous features?
    
</question>

<answer>

To avoid having to check the optimal split for continuous features at every split, we could bin these variables while preprocessing our data. This would save us a lot of computation, but would likely not result in as good of a rule set.

</answer>

</qinline>

### Citations
- https://en.wikipedia.org/wiki/Entropy
