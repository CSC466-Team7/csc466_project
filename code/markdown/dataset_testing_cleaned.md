```python
import pandas as pd
import numpy as np
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
target_col = 'target'
```


```python
y.value_counts()
```




    1    165
    0    138
    Name: target, dtype: int64




```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X = heart_disease.drop(columns=[target_col])
y = heart_disease[target_col]

clf = DecisionTreeClassifier(random_state=0)
cross_val_score(clf, X, y, cv=10)
```




    array([0.80645161, 0.83870968, 0.90322581, 0.86666667, 0.76666667,
           0.7       , 0.7       , 0.73333333, 0.7       , 0.8       ])




```python
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

t = y

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25,random_state=0)

model = clf.fit(X_train, t_train)

feature_names = X.columns

r = permutation_importance(model, X_test, t_test, n_repeats=30, random_state=0)
# print(r)
for i in r.importances_mean.argsort()[::-1]:
#      if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    print(f"{feature_names[i]:<8}"
          f"{r.importances_mean[i]:.3f}"
          f" +/- {r.importances_std[i]:.3f}")
```

    cp      0.109 +/- 0.034
    ca      0.043 +/- 0.023
    thal    0.040 +/- 0.019
    restecg 0.014 +/- 0.007
    chol    0.013 +/- 0.020
    exang   0.012 +/- 0.010
    age     0.011 +/- 0.022
    oldpeak 0.011 +/- 0.020
    sex     0.001 +/- 0.019
    slope   -0.001 +/- 0.006
    fbs     -0.002 +/- 0.004
    trestbps-0.005 +/- 0.011
    thalach -0.006 +/- 0.030

