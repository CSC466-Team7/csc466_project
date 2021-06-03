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

### We will be implementing a decision tree that uses entropy to determine information gain for deciding when to split. If confused, check out the [introduction on decision trees](https://csc466-team7.github.io/csc466_project/#/introduction)

### Entropy

Entropy is most commonly associated with a state of disorder, randomness, or uncertainty. Here we use it to find how much our target column is different. Our formula for entropy is:

${\displaystyle \mathrm {H} {(S)}=\sum _{y\in V_Y}{-p(y)\log _{2}p(y)}}$

where `H(S)` is the entropy and `p(y)` is the proportion of occurrences of a given class in y.

If we have only 1 class in `y`, then there is no entropy or disorder so the entropy would be 0. We can see this because `p(y) = 1` so we'd have

${\displaystyle \mathrm {H} {(S)}={-(1)\log _{2}(1)}}={-1 * 0}={0}$

If we have more than 1 class, then we have at least some disorder in our data. If we had only 2 classes of say infinite values and 1 was *Class A* and all others were *Class B* we'd have very low entropy (approaching 0) since nearly all are the same class. Our entropy would be the worst if we had **half and half** between the 2 classes. If we have more classes, say `n`, the worst entropy involves having `1/n` as the proportion of each class in `y`.


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

<qinline>

<question>

Why bother taking the ${\displaystyle \mathrm -log _{2}(p(y))}$? Why not just sum the proportion of that class?

</question>

<answer>

- For starters, if we just took ${\displaystyle \mathrm p(y)}$ in the sum, the sum would always equal 1. So that doesn't tell is how much randomness we have!
- We still want to be able to say only 1 class means 0 entropy. This means if ${\displaystyle \mathrm p(y) = 1 -> H(S) = 0}$. This is satisfied as a property of logarithms!
- We also want to say equal probabilties across all classes means ${\displaystyle \mathrm H(S) = 1}$. This means if we randomly selected 1 observation, it could be any of our `n` classes, which means there is maximum uncertainty for which would be chose. `1` is a convenient number for showing that. With logarithms of any base, this is satisfied with the equation.
- The negative sign at the beginning of the equation ensures that as probability increases, our entropy decreases.
- Thus, using ${\displaystyle \mathrm -log _{2}(p(y))}$ seems to satisfy all of what we want. In all, the logarithm part is what actually tells us the uncertainty of that class. Multiplying by the proportion rescales it for summing later.
- For more information, I recommend visiting [here](https://stats.stackexchange.com/questions/87182/what-is-the-role-of-the-logarithm-in-shannons-entropy#:~:text=so%20the%20properties%20of%20logarithm,entropy%20for%20%CE%B1%E2%86%921\).&text=as%20the%20average%20of%20log,(1%2Fp)%20information.) 

</answer>

</qinline>

### Information Gain

Entropy tells us, given the target values that are left, how much disorder there is within them. This is a good start. But let's say we are at a node of the tree, `N`, and we are deciding between choosing feature `A` and feature `B` for our next branch of the tree to use. How do we do that?

One way is to develop a method to see how much information is gained if we choose feature `A` and how much is gained if we choose feature `B`. We can do this by going through each feature we have left (so `A` and `B` in this case) and finding how much we learn for the related targets. Luckily we have half of what we need to do this already since we now know about *entropy*!

Let's say we arbitrarily are looking at feature `B`. Feature `B` has 3 classes, `X`, `Y`, and `Z`. Let's look at `X` first.

First we need to find the targets related to where the value of `B = X`. This will be a subset of our current targets, assuming `Y` and `Z` both have at least 1 observation. Once again, if `X`'s targets all consist of only 1 value, we'd say it has very low entropy. We can find how good our entropy is proportional to the other classes by finding the proportion of times `B = X` and multiplying that by the entropy

${\displaystyle \mathrm Proportional Entropy={p(B = X) * Entropy{(Targets{(B = X)})}}}$

We take the proportion of each class within `B` because we care more about the entropy of a class within `B` that has *20x* the number of observations than any other class in `B`. This gives us a relative gain for that class proportional to the other classes.

From there, we sum our gains together to find the total entropy for that feature. So how much did we learn if we used feature `B`? Well, we can take our previous entropy before trying to use `B` has our next split and subtract the summed entropy of each each class in `B` from that.

${\displaystyle \mathrm Gain=OriginalEntropy-{\sum _{c\in B_C}}{Proportional Entropy(c)}}$

The summed entropy must be less than or equal to the original entropy. For this reason, if our gain is 0, we can stop, since it will never get better.

<qinline>

<question>

If `Class A` has entropy `0.9` and `Class B` has entropy `0.6` and the entropy disregarding the classes was `1.4`, which class has higher information gain?

</question>

<answer>

`Class B`
<br>
If we started with an entropy of `1.4` and decreased it to `0.6`, our entropy decreased by `0.8`. For `Class A`, it only decreased by `0.5`. That means we got more information (can make better predictions at this point according to the training data) from `Class B`, since the difference in entropy was higher. 

</answer>

</qinline>


```python
# Calculates information gain of a given feature
def gain(y,x):
    g = 0
    for v in x.unique():
        sub_t = y.iloc[np.where(x == v)]
        g += (len(sub_t) / len(y)) * entropy(sub_t)
    return entropy(y) - g
```

### Selecting the Best Split

Now that we have a way to determine the information gain of any given feature, we can go ahead and select the best feature at an arbitrary point in our decision tree. We just need to loop over all features and select the one with the highest gain.


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
```

## Creating our Tree

Now we need to put it all together to make our decision tree. Since trees are naturally recursive, the easiest way to write this is recursively. First, let's discuss our base cases, or when we can stop making new additions to our tree.

### Base Cases

1. The number of unique classes left in the target is 1. *This means that the entropy of target is 0* so we know what selection to make... the only value in the target left.

2. There are no more features. Let's say we have 1 more feature left and so we must select that one. If we have anything left over after that, we have nothing left to split on. Thus, we just make a logical decision and select the class in the target that occurs most often.

<qinline>

<question>

1. If `Class A` of `Feature 1` has an entropy of `0.3` and `Class X` of `Feature 3` has an entropy of `0.6`, which feature will be selected as the best split? Do we know that for sure?
2. Assuming the 2 features had the same original entropy, can we know decide what our best split is between the 2 classes presented (assume no other class of either feature will be better)? If so, which? If not, why?

</question>

<answer>

1. We don't know. We'd need to know the original entropies of both `Feature 1` and `Feature 3` to decide.
2. We do now know. `Feature 1`, since it had an lower entropy, must have had a larger decrease in entropy, meaning higher information gained.

</answer>

</qinline>


```python
def tree_creation(X, y):
    if len(y.unique()) == 1:
        return y.iloc[0]
    
    if len(X.columns) == 0:
        return high_freq_class(y)
    
    col, gr = select_split(X, y)
    
    return tree_creation_main(X, y, col)
```

### Recursive Part

At this point, if we haven't hit a base case, we know we've found a valid split for our tree. This means we know want to split our tree into each class of the feature we found. After that, we need to make a new tree with the features left after the selection of this given feature. So the steps are:

- Go through each unique class, `c`, in the feature, `B`
  - Create a new observation list where the observations (targets and other features) have `B = c` in that observation
        indexes = where(B = c)
        new_X = X.locate_by_index[indexes]
        new_y = y.locate_by_index[indexes]

  - Remove feature `B` from our observations
      ```
      new_X.drop(columns=[col])
      ```
  - Add to the tree at feature `B` class `c` whose value is a new tree
     ```
     tree[B][c] = tree(new_X, new_y)
     ```


```python
# Creates actual decision tree
def tree_creation_main(X,y,col):
    tree = {col: {}}

    unique_vals = X[col].unique()
    
    for v in unique_vals:
        indexes = np.where(X[col] == v)
        new_X = X.iloc[indexes].drop(columns=[col])
        new_y = y.iloc[indexes]
        tree[col][str(v)] = tree_creation(new_X, new_y)

    return tree
```

<qinline>

<question>

Is it possible to generate a tree with no features in it when given a dataset with 5 features with 30 observations? Why?

</question>

<answer>

Yes, if there is only one target class it doesn't matter. That's always what you'd want to predict!

</answer>

</qinline>

### Generating the Rules

Now that we have a complete tree, we need to be able to make a new prediction with it. We will do that by following our tree down the branches we have made. If we want to visualize it, we can imagine that each path from the root of our tree down to the leaves of our tree are a rule for how to predict what a given input should output.

We can generate a list of all rules by doing a full traversal of the tree. If we hit a leaf, we are at the end of our tree and just return that leaf as the only rule. Otherwise, for each class of a given feature in our tree, we need to make a new rule. This means that each rule should start with the root node and after that may have differences to other rules. Let's refer to the 
[introduction on decision trees](https://csc466-team7.github.io/csc466_project/#/introduction).


Our root node is the feature `Weather`. That has 2 classes, `warm` and `cold`. Then all rules would start with either `Weather = Cold` or `Weather = Warm`. Let's follow the left hand side of the tree. The next part of the rule would be the `Day of the week`. We would make 2 new rules for `Day of the week = weekday` and `Day of the week = weekend`. If we had a prediction with `Weather = Cold, Day of the week = weekday`, then we would want our rule to say that we'd use a `T-shirt`. Remember, all the rules together are just each unique path from the root node to a leaf.


```
feature = tree

if tree.is_leaf():
    return tree
else:
    for class in feature:
        branch_step = [feature, class]
        return branch_step + find_new_rules(feature[class])
```

Once we are done, we should have a dictionary or list of rules. Understanding how rules get generated and how to use them to make predictions is important.

<qinline>

<question>

Do you actually need to generate a ruleset? Can you just make predictions instead using the tree itself?

</question>

<answer>
    
Absolutely! The ruleset is a nice visualization of what's going on and shows you the different paths your tree can take. It *may* also make the coding a little easier for the predictions. However, you can certainly use just the tree instead moving forward. After all, the ruleset was just generated from the tree.

</answer>

</qinline>


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
```

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

<qinline>

<question>

Why give a default? The use-case above has an example of when to return it but can you give an example of how it might make it there? Let's say you have the following ruleset:
```
[[(Gender, Male), 1],
 [(Gender, Female), 0]]
```

Give a possible input as `x`, your observation, for getting to the default base-case.

</question>

<answer>

Possible observation:
- (Gender = Non-Binary)
- (Laugh = Loud)

You can either give no classes that match OR you can give a `Gender` that doesn't exist.

</answer>

</qinline>


```python
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
    print(matching_value)

    viable_rules = list(filter(filter_rule, rules))
    
    if len(viable_rules) == 0:
        return default
    
    new_rules = list(map(lambda x: x[1:], viable_rules))

    return make_prediction(new_rules, x, default)
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


# Not bad!

<qinline>

<question>

Why do you think our custom tree might have done better than SciKit-Learn's implementation?

</question>

<answer>

This actually has to do with at least 2 things: when we decide to stop with building our tree (if we return when the infromation gain of the best split is 0) and how we decide which class from the target to choose if there is equal amount of both. If we vary those 2 things, our validation may be better. This is more luck based on the dataset then it is practically better. Can you think of ways to optimize this behavior? 

</answer>

</qinline>

### Citations
- https://en.wikipedia.org/wiki/Entropy
