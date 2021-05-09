
# Predicting Heart Disease


```python
import numpy as np
import pandas as pd
import copy
import json
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
```

## Read in Data


```python
# https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv

heart_disease = pd.read_csv("../../datasets/heart.csv")
heart_disease.head()
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
heart_disease['thal'].unique()
```




    array([1, 2, 3, 0], dtype=int64)



## Brief Cleaning


```python
heart_disease = heart_disease.rename(columns={'target': 'disease_present'})
```


```python
target_col = 'disease_present'
# Remove quantitative variables -- for now
other_to_drop = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
t = heart_disease['disease_present']
X2 = heart_disease.drop(columns=([target_col] + other_to_drop))
```

## Setting up Decision Tree (for only categorical features/target for now)

### We will be implementing a decision tree that uses entropy to determine information gain for deciding when to split. If confused, check out the [introduction on decision trees]()


```python
# Calculates the entropy that exists in Series.
def entropy(y):
    e = 0
    for v in y.unique():
        p_v = np.sum(y == v) / len(y)
        total = -1 * (p_v * np.log2(p_v))
        e += total
    return e
```


```python
# Calculates information gain of a given feature
def gain(y,x):
    g = 0
    for v in x.unique():
        sub_t = y.iloc[np.where(x == v)]
        g += (len(sub_t) / len(y)) * entropy(sub_t)
    return entropy(y) - g
```


```python
# Finds the optimal feature and information gain of dataset X
def select_split(X,y):
    col = None
    gr = -1
    for c in X.columns:
        cur_gain_ratio = gain(y, X[c])
        if cur_gain_ratio > gr:
            gr = cur_gain_ratio
            col = c
    return col,gr
```


```python
# Finds the class with the highest frequency
def high_freq_class(y):
    v_counts = y.value_counts()
    return v_counts.idxmax()

# Creates actual decision tree
def make_tree(X,y):
    if len(y.unique()) == 1:
        return y.iloc[0]
    
    if len(X.columns) == 0:
        return high_freq_class(y)
    
    tree = {}
    col, gr = select_split(X, y)
    tree[col] = {}
    
    if gr == 0:
        return high_freq_class(y)
    
    unique_vals = X[col].unique()
    
    for v in unique_vals:
        indexes = np.where(X[col] == v)
        new_X = X.iloc[indexes].drop(columns=[col])
        new_y = y.iloc[indexes]
        tree[col][str(v)] = make_tree(new_X, new_y)

    return tree
```


```python
# Takes a tree and generates the rules of that tree used to make a prediction
def generate_rules(tree):
    rules = []
    if type(tree) != dict:
        return [[tree]]
    for col in tree:
        for val in tree[col]:
            tup = (col, val)
            generated_sub_rules = generate_rules(tree[col][val])
            for sub_rule in generated_sub_rules:
                new_rule = [tup]
                new_rule.extend(sub_rule)
                rules.append(new_rule)
    
    return rules
```


```python
# Returns a predicate of whether a given values matches a given rule's first feature value 
def eq_rule(val_to_match):
    return lambda x: x[0][1] == str(val_to_match)

# Used to make a prediction given a decision tree's rule and some inputs
def make_prediction(rules,x,default):
    if len(rules) == 0:
        return default
    
    tups = []
    next_rule = rules[0][0]

    if type(next_rule) != tuple:
        return next_rule
    
    col = next_rule[0]
    
    matching_value = x[col]
    filter_rule = eq_rule(matching_value)

    viable_rules = list(filter(filter_rule, rules))
    
    if len(viable_rules) == 0:
        return default
    
    new_rules = list(map(lambda x: x[1:], viable_rules))

    return make_prediction(new_rules, x, default)
```


```python
# Implementation of C4.5
def make_tree2(X,y,min_split_count=5):
    if len(y.unique()) == 1:
        return y.iloc[0]
    
    if len(X.columns) == 0:
        return high_freq_class(y)
    
    num_elements = len(y)
    
    if num_elements < min_split_count:
        return high_freq_class(y)
    
    tree = {}
    # Your solution here
    col, gr = select_split(X, y)
    tree[col] = {}
    
    if gr <= 0:
        return high_freq_class(y)
        
    for v in X[col].unique():
        indexes = np.where(X[col] == v)
        new_X = X.iloc[indexes].drop(columns=([col]))
        new_y = y.iloc[indexes]
        tree[col][str(v)] = make_tree2(new_X, new_y, min_split_count)

    return tree
```


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

## Predict

- We will use sklearn's `train_test_split` function to break up the dataset into a training set and test set


```python
default = 0
from sklearn.model_selection import train_test_split

X2_train, X2_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state = 0)

tree = make_tree(X2_train,t_train)
rules = generate_rules(tree)

y_test = X2_test.apply(lambda x: make_prediction(rules,x,default),axis=1)
```


```python
print_tree(tree)
```

    {
        "thal": {
            "0": {
                "sex": {
                    "0": 1,
                    "1": 0
                }
            },
            "1": {
                "ca": {
                    "0": {
                        "restecg": {
                            "0": 1,
                            "1": {
                                "slope": {
                                    "0": 0,
                                    "1": 1
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
                "ca": {
                    "0": {
                        "restecg": {
                            "0": {
                                "cp": {
                                    "0": {
                                        "slope": {
                                            "1": 1,
                                            "2": {
                                                "sex": {
                                                    "0": 1,
                                                    "1": 1
                                                }
                                            }
                                        }
                                    },
                                    "1": {
                                        "sex": {
                                            "0": 1,
                                            "1": {
                                                "slope": {
                                                    "1": 1,
                                                    "2": 1
                                                }
                                            }
                                        }
                                    },
                                    "2": 1,
                                    "3": {
                                        "sex": {
                                            "0": 1,
                                            "1": {
                                                "exang": {
                                                    "0": 1,
                                                    "1": 1
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "1": {
                                "cp": {
                                    "0": {
                                        "exang": {
                                            "0": 1,
                                            "1": {
                                                "slope": {
                                                    "1": 0,
                                                    "2": 1
                                                }
                                            }
                                        }
                                    },
                                    "1": 1,
                                    "2": {
                                        "sex": {
                                            "0": 1,
                                            "1": {
                                                "exang": {
                                                    "0": {
                                                        "slope": {
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
                                "cp": {
                                    "0": 0,
                                    "2": 1
                                }
                            }
                        }
                    },
                    "1": {
                        "cp": {
                            "0": 0,
                            "1": 1,
                            "2": 1,
                            "3": 1
                        }
                    },
                    "2": {
                        "exang": {
                            "0": {
                                "fbs": {
                                    "0": {
                                        "sex": {
                                            "0": 1,
                                            "1": {
                                                "restecg": {
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
                        "cp": {
                            "0": 0,
                            "2": {
                                "fbs": {
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
                "cp": {
                    "0": {
                        "ca": {
                            "0": {
                                "slope": {
                                    "0": 0,
                                    "1": 0,
                                    "2": {
                                        "restecg": {
                                            "0": {
                                                "exang": {
                                                    "0": 0,
                                                    "1": 0
                                                }
                                            },
                                            "1": {
                                                "exang": {
                                                    "0": 1,
                                                    "1": 1
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "1": {
                                "slope": {
                                    "1": {
                                        "restecg": {
                                            "0": 0,
                                            "1": {
                                                "exang": {
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
                        "slope": {
                            "0": 0,
                            "1": 1,
                            "2": {
                                "fbs": {
                                    "0": {
                                        "restecg": {
                                            "0": 1,
                                            "1": 1
                                        }
                                    },
                                    "1": 1
                                }
                            }
                        }
                    },
                    "2": {
                        "slope": {
                            "1": {
                                "ca": {
                                    "0": {
                                        "exang": {
                                            "0": 1,
                                            "1": {
                                                "fbs": {
                                                    "0": {
                                                        "restecg": {
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
                        "fbs": {
                            "0": {
                                "restecg": {
                                    "0": 1,
                                    "1": {
                                        "slope": {
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
    

### How did our decision tree do with the test set?


```python
from sklearn.metrics import accuracy_score, f1_score

print(f'Accuracy: {accuracy_score(y_test, t_test)}')
print(f'F1 score: {f1_score(y_test, t_test)}')
```

    Accuracy: 0.8241758241758241
    F1 score: 0.8367346938775511
    

### How does this compare with the battle-tried SciKit Learn version?


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=0,criterion='entropy',min_samples_split = 2)
model = clf.fit(X2_train, t_train)

y_model_test = model.predict(X2_test)

print(f'Accuracy: {accuracy_score(y_model_test, t_test)}')
print(f'F1 score: {f1_score(y_model_test, t_test)}')
```

    Accuracy: 0.7912087912087912
    F1 score: 0.7999999999999999
    

### Not bad!
