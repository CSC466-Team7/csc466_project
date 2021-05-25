```python
import numpy as np
import pandas as pd
import copy
import json
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
```


```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels, check_classification_targets
```


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
heart_disease = heart_disease.rename(columns={'target': 'disease_present'})
```


```python
target_col = 'disease_present'
# Remove quantitative variables -- for now
other_to_drop = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
t = heart_disease['disease_present']
X2 = heart_disease.drop(columns=([target_col] + other_to_drop))
```


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
    
    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.cols_ = X.columns
        else:
            self.cols_ = None
        
        X, y = check_X_y(X, y)
        
        check_classification_targets(y)
        
        self.classes_ = np.unique(y)
        
        self.X_ = X
        self.y_ = y
        
        # https://github.com/scikit-learn/scikit-learn/blob/053d2d1af477d9dc17e69162b9f2298c0fda5905/sklearn/tree/_classes.py#L83

        X = np.copy(X)
        y = np.copy(y)

        n_samples, self.n_features_in_ = X.shape
        
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.n_outputs_ = 1

        start_idxs = np.asarray(list(range(self.n_features_in_)))
        self._tree = self._make_tree(X, y, start_idxs)
        self._rules = self._get_rules(self._tree)

        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Features in predict different than features in fit")
        
        default = self.y_[0] if self.default == None else self.default
        
        y_vals = []
        for x in X:
            y_vals.append(self._make_prediction(self._rules,x,default))
        
        return np.asarray(y_vals)

    def _get_entropy(self, y):
        e = 0
        for v in np.unique(y):
            p_v = np.sum(y == v) / len(y)
            total = -1 * (p_v * np.log2(p_v))
            e += total
        return e
    
    def _gain(self, y,x):
        g = 0
        for v in np.unique(x):
            sub_t = y[np.where(x == v)]
            g += (len(sub_t) / len(y)) * self._get_entropy(sub_t)
        return self._get_entropy(y) - g
        
    
    # use counts in case of tie
    def _high_freq_class(self, y):
        # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
#         y_counts = np.bincount(y)
#         y_vals = np.where(y_counts == y_counts.max())[0]
#         numpy.argsort(vals)
        u, c = np.unique(y, return_counts = True)
        temp = u[c == c.max()]
        return temp[0]

    def _make_tree(self,X,y,related_idxs):
        if len(np.unique(y)) == 1:
            return y[0]

        if X.shape[1] == 0:
            return self._high_freq_class(y)

        tree = {}
        
        # Find best split
        col = None
        gr = -1
        for c in range(X.shape[1]):
            cur_gain_ratio = self._gain(y, X[:, c])
            if cur_gain_ratio > gr:
                gr = cur_gain_ratio
                col = c
    
    
        correct_col = related_idxs[col]
        tree[correct_col] = {}

        if gr == 0:
            return self._high_freq_class(y)
        
        X_col = X[:, col]
        unique_vals = np.unique(X_col)

        for v in unique_vals:
            assert(X_col.ndim == 1)
            indexes = np.where(X_col == v)
            new_X = X[indexes[0], :]
            new_X = np.delete(new_X, col, axis=1)
            new_y = y[indexes]
            new_valid_idxs = np.delete(related_idxs, col, axis=0)
            tree[correct_col][str(v)] = self._make_tree(new_X,new_y,new_valid_idxs)

        return tree
    
    def _get_rules(self, tree):
        rules = []
        if type(tree) != dict:
            return [[tree]]
        for col in tree:
            for val in tree[col]:
                tup = (col, val)
                generated_sub_rules = self._get_rules(tree[col][val])
                for sub_rule in generated_sub_rules:
                    new_rule = [tup]
                    new_rule.extend(sub_rule)
                    rules.append(new_rule)

        return rules

    def _eq_rule(self, val_to_match):
        def eq_matcher(x):
            return x[0][1] == str(val_to_match)
        
        return eq_matcher

    # Used to make a prediction given a decision tree's rule and some inputs
    def _make_prediction(self, rules,x,default):
        if len(rules) == 0:
            return default

        tups = []
        next_rule = rules[0][0]

        if type(next_rule) != tuple:
            return next_rule

        col = next_rule[0]

        matching_value = x[col]
        filter_rule = self._eq_rule(matching_value)

        viable_rules = list(filter(filter_rule, rules))

        if len(viable_rules) == 0:
            return default

        new_rules = list(map(lambda x: x[1:], viable_rules))

        return self._make_prediction(new_rules, x, default)
    
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


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X2_train, X2_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state = 0)

clf = CustomDecisionTreeClassifier(default=1)
y_test = clf.fit(X2_train, t_train).predict(X2_test)

print(f'Accuracy: {accuracy_score(y_test, t_test)}')
print(f'F1 score: {f1_score(y_test, t_test)}')
```

    Accuracy: 0.7912087912087912
    F1 score: 0.7999999999999999



```python
from sklearn.utils.estimator_checks import check_estimator
from sklearn.tree import DecisionTreeClassifier
check_estimator(CustomDecisionTreeClassifier())
```
