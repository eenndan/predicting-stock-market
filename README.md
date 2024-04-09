# Machine Learning Project

## Daniil Ennus

## London Business School 

## Machine Learning for Big Data Course

The purpose of this project is to build a prediction model to guide
trading decisions in stock market given variety of information.


```python
# Import liraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
```


```python
#%% import data

# import both datasets
filename = 'design_matrix.csv'
X = pd.read_csv(filename)


filename = 'target.csv'
y = pd.read_csv(filename).squeeze()


# Remove redundant variables
del filename
```


```python
'''

Number of columns is enourmous and I suspect that many of those will
have very limited useful information. Thus, it is logical to 
reduce dimensionality using PCA.

But to make final decision, let's first inspect the data

'''

# Calculate average values for each column
average_values = X.mean()

# Filter average values below 5
average_values = average_values[average_values < 5]

# Plot histogram for the distribution of average values across all columns
plt.figure(figsize=(8, 6))
plt.hist(average_values, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Average Values Across All Columns')
plt.xlabel('Average Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

'''
This gives slightly better understanding of data
'''

del average_values
```


    
![png](output_3_0.png)
    



```python
# Count the number of columns with binary values [0, 1]
num_binary_columns = ((X == 0) | (X == 1)).sum()

# Filter the columns where the count is equal to the number of rows
binary_columns = num_binary_columns[num_binary_columns == len(X)]

# Get the number of columns with binary values [0, 1]
print(len(binary_columns))
# 33 binary columns
```

    33



```python
'''
It is incorrect so shuffle the data as it can generate
data leakages. The mistake is even more serious in model which
relies on past observations. The entire logic of such model is 
ruined if data is shuffled.


Also, one can argue that market microstructure has changed a lot in 
the recent years mostly due to algo trading. Thus, the markets today
and 20 years ago are very different. Hence, there is no point studying
such a long timeframe, and, for example, 5 years look reasonable. 
'''

# Split data. Cannot randomly shuffle as sequence is braking
# Last n years for training and validation. Assuming 1 years is 252 trading days
# Create function to make it easier change timeframe
# Considering that the newest data is at the bottom of the dataframe

def split(n_years):
    X_test = X[-65:]  # Selecting the last 65 days for testing
    y_test = y[-65:]  # Corresponding labels for testing
    
    
    for_train = round(0.8 * n_years * 252 + 65) # 80% of the selected timeframe
    for_val = round(0.2 * n_years * 252 + for_train) # remaining 20%

    X_train = X[-for_train: -65]  # Selecting from the end, excluding the last 65 days for training
    X_val = X[-for_val: -for_train]   # Selecting from the end for validation

    
    y_train = y[-for_train: -65]
    y_val = y[-for_val: -for_train] 
    
    return X_test, X_train, X_val, y_test, y_train, y_val

X_test, X_train, X_val, y_test, y_train, y_val = split(5)

```


```python
#%% SVM 5 years

X_test, X_train, X_val, y_test, y_train, y_val = split(5)

random_seed = 124

# Scale data. We have different units. Thus, we need to scale
# using both mean and stdev
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define the SVM classifier and parameter grid for tuning
svm = SVC(kernel='rbf', 
          random_state = random_seed)  # Radial basis function kernel

param_grid = {'C': [0.1, 1, 2, 5, 7, 10], 'gamma': [1, 0.1, 0.01, 0.005, 0.001]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10)
grid_search.fit(X_train_scaled, y_train)

# Get the best SVM model
best_svm_model = grid_search.best_estimator_
best_svm_model

```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(C=0.1, gamma=0.001, random_state=124)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(C=0.1, gamma=0.001, random_state=124)</pre></div></div></div></div></div>




```python
# All results
results_df = pd.DataFrame(grid_search.cv_results_)[["param_C", "param_gamma", "mean_test_score"]]
results_df

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
      <th>param_C</th>
      <th>param_gamma</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1</td>
      <td>0.01</td>
      <td>0.666644</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1</td>
      <td>0.005</td>
      <td>0.704416</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1</td>
      <td>0.001</td>
      <td>0.711050</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.1</td>
      <td>0.488089</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.01</td>
      <td>0.662604</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0.005</td>
      <td>0.659644</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.001</td>
      <td>0.651723</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>0.1</td>
      <td>0.484129</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>0.01</td>
      <td>0.662604</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>0.005</td>
      <td>0.657644</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>0.001</td>
      <td>0.646713</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>0.1</td>
      <td>0.485119</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>0.01</td>
      <td>0.652703</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>0.005</td>
      <td>0.657653</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5</td>
      <td>0.001</td>
      <td>0.659634</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>0.1</td>
      <td>0.485119</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>0.01</td>
      <td>0.640822</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>0.005</td>
      <td>0.662624</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7</td>
      <td>0.001</td>
      <td>0.664564</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10</td>
      <td>1</td>
      <td>0.503960</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10</td>
      <td>0.1</td>
      <td>0.485119</td>
    </tr>
    <tr>
      <th>27</th>
      <td>10</td>
      <td>0.01</td>
      <td>0.640822</td>
    </tr>
    <tr>
      <th>28</th>
      <td>10</td>
      <td>0.005</td>
      <td>0.661644</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10</td>
      <td>0.001</td>
      <td>0.665535</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot results
for gamma, group_df in results_df.groupby('param_gamma'):
    plt.plot(group_df['param_C'], group_df['mean_test_score'], label=f'Gamma: {gamma}')

plt.xlabel('C')
plt.ylabel('Mean Test Score')
plt.title('Mean Test Score vs. C for different Gamma values')
plt.legend(title='Gamma')
plt.grid(True)
plt.show()

```


    
![png](output_8_0.png)
    



```python

```
