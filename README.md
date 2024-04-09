# Machine Learning Project

### Daniil Ennus

### London Business School 

## Machine Learning for Big Data Course

The purpose of this project is to build a prediction model to guide
trading decisions in stock market given variety of information.

### Table of contetns:
1. Exploring data
2. SVM on entire data
3. PCA
4. SVM on PCA-reduced data
5. ANN model
6. LSTM model
7. Final model

# Exploring data


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

del average_values
```


    
![png](output_4_0.png)
    


## Identify binary columns to later remove for PCA


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

# SVM model on entire data


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


    
![png](output_11_0.png)
    



```python
# Confussion matrix
X_val_scaled = scaler.transform(X_val) # validation data is transformed 
        #using mean and stdev of training data to eliminate leakages

y_pred = grid_search.best_estimator_.predict(X_val_scaled) # Get predictions from the best model
conf_matrix = confusion_matrix(y_val, y_pred) # create condusion matrix
conf_matrix
```




    array([[ 60,   1],
           [146,  45]])




```python
accuracy = np.mean(y_val == y_pred)
accuracy # 0.42
```




    0.4166666666666667




```python
# Now let's adjust decision threshold
# Decision scores
decision_scores = grid_search.best_estimator_.decision_function(X_val_scaled)
```


```python
# Set custom threshold
custom_threshold = -0.1  # Adjust as needed

# Make predictions based on custom threshold
y_pred = (decision_scores > custom_threshold).astype(int)
conf_matrix = confusion_matrix(y_val, y_pred)
conf_matrix

accuracy = np.mean(y_val == y_pred)
accuracy # 0.56


disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["No", "Yes"])
disp.plot()

```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb2930be320>




    
![png](output_15_1.png)
    



```python
# AUC and ROC
# Calculate false positive rate (fpr) and true positive rate (tpr) for various thresholds
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, decision_scores)

# Calculate the Area Under Curve (AUC)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_val, decision_scores)
auc # 0.78

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='skyblue', label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](output_16_0.png)
    


# PCA


```python
#%% PCA. Identifying number of PCs and performing data

'''
Now let's run PCA to reduce dimensionality. 500+ variables is definitely too much,
especially considering that we are only dealing with 5 years of data.

Dummy variables may be a problem for PCA. Thus, I will split data, run PCA,
and then join dummies back. 

I will first first scale data (as well as dummies) as study materials mostly 
recommend to scale all variables if SVM is with a kernel, which is my case.  
'''

X_test, X_train, X_val, y_test, y_train, y_val = split(5)

# Shape of the data
X_train.shape

# Creating a logical vector where True represents columns with a sum of 0
logical_vec = X_train.sum() == 0

# Getting the index numbers corresponding to the True values in the logical vector
res = np.where(logical_vec)[0]

# Printing the result
print(res) # no columns perfectly empty


# Calculate proportion of each value in each column
value_counts = X_train.apply(lambda x: x.value_counts(normalize=True))

# Filter out columns where one value dominates the majority.
dominant_value_threshold = 0.7  # Adjust as needed.
dominant_columns = value_counts.columns[(value_counts.max(axis=0) > dominant_value_threshold)]

# Print the dominant columns
print("Columns where one value dominates the majority:")
print(dominant_columns)
```

    []
    Columns where one value dominates the majority:
    Index(['20', '55', '56', '59', '61', '91', '135', '180', '320', '491', '515',
           '517', '519', '530', '531'],
          dtype='object')



```python
# Now let's exclude these columns with value dominating 70%
filtered_X_train = X_train.drop(columns=dominant_columns)

# Center the data first and  scale
scaler = StandardScaler(with_mean=True, with_std=True)
X_centered = scaler.fit_transform(filtered_X_train)

# Now remove dummy columns
binary_columns = [int(x) for x in binary_columns.index]

# Split data to binary and continuous
X_train_cont = pd.DataFrame(X_centered).drop(columns = binary_columns)
X_train_bin = pd.DataFrame(X_centered)[binary_columns]
```


```python
# Perform PCA  
pca = PCA()
pca.fit(X_train_cont)
```


```python
# 1) Basis Vectors (eigenvectors) 
basis_vectors = pca.components_
print('Basis Vectors Information')
print(pd.DataFrame(basis_vectors[:3, :3], columns=['PC1','PC2','PC3'])) # First 3 loadings for the first 3 directions
print(pca.components_.shape)   # Shape of the basis matrix
print('\n')

```

    Basis Vectors Information
            PC1       PC2       PC3
    0 -0.008547  0.009898  0.009701
    1  0.009574  0.007715 -0.000776
    2  0.034375 -0.001446  0.010816
    (510, 510)
    
    



```python
# 2) Scores (new points in the principal component space)
print('Scores Information') 
Z = pca.transform(X_train_cont)  # Transform the X_scaled inputs to the principal component space
print(pd.DataFrame(Z[:3, :3], columns=['PC1','PC2','PC3']))  # First 3 rows and first 3 columns of the transformed data
print(Z.shape)   # Shape of the transformed data
print('\n')
```

    Scores Information
            PC1       PC2       PC3
    0  6.734421 -0.405879 -4.837758
    1  6.570937 -0.188038 -4.367345
    2  6.366504  0.041746 -4.330476
    (1008, 510)
    
    



```python
# To calculate the Proportion of Variance Explained (PVE) and Cumulative PVE
print('Variance Analysis')
ve = pca.explained_variance_
pve = pca.explained_variance_ratio_ * 100
cpve = np.cumsum(pve)

# Let's display the first 5 of each PC's PVE for comparison
print("First 5 PVE:", pve[:5].round(2))
print("First 5 CPVE:", cpve[:5].round(2))
```

    Variance Analysis
    First 5 PVE: [23.26 17.62 12.32  5.4   4.72]
    First 5 CPVE: [23.26 40.87 53.2  58.6  63.32]



```python
# Plot data
N, D = X_train_cont.shape  # N is the number of observations, D is the number of features

# Create a DataFrame
df = pd.DataFrame({'PC': np.arange(1, D + 1),
                   'var_explained': ve,
                   'cum_sum_PVE': cpve})

# Select the first M principal components for the plot
M = 20
df_subset = df.iloc[:M]

# Plotting importance of PCs
plt.figure(figsize=(10, 6))
plt.scatter(df_subset['PC'], df_subset['var_explained'], s=50)
plt.plot(df_subset['PC'], df_subset['var_explained'], linestyle='-')
plt.xticks(np.arange(1, M + 1))
plt.xlabel('PC Number', fontsize=15)
plt.ylabel('VE', fontsize=15)
plt.title('Scree Plot for PCA on MNIST', fontsize=18)
plt.grid(True)
plt.show()
```


    
![png](output_24_0.png)
    



```python

# Cumulative Proportion of Variance Explained By Each PC
# Plotting CPVE for the first M principal components
M = 65
df_subset = df.iloc[:M]


'''
90% of variance is explained by 25 PCs. Whhich suggests that a lot of data is 
redundant. This observation is also supported by very strong correlation 
between variables seen previously.
Let's keep even lower number of PCs - only 20
'''


plt.figure(figsize=(10, 6))
plt.scatter(df_subset['PC'], df_subset['cum_sum_PVE'], color='green', s=50)
plt.ylim(0, 100)
plt.axhline(y=90, linestyle='--', color='red', linewidth=1.2)
plt.xlabel('PC', fontsize=15)
plt.ylabel('CPVE', fontsize=15)
plt.title('Cumulative Proportion Variance Explained Plot - PCA on MNIST Data', fontsize=18)
plt.grid(True)
plt.show()

```


    
![png](output_25_0.png)
    



```python
# Keep only 20 most valuable PCs
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_cont)

# Check that there are only 20 columns
X_train_pca.shape

# Now let's join dummies back
X_train_pca = pd.concat([pd.DataFrame(X_train_pca), X_train_bin], axis = 1)
```

# SVM on PCA-reduced data


```python

# Run SVM again
random_seed = 124

# Define the SVM classifier and parameter grid for tuning
svm = SVC(kernel='rbf', 
          random_state = random_seed)  # Radial basis function kernel

param_grid = {'C': [0.1, 1, 2, 5, 7, 10], 'gamma': [1, 0.1, 0.01, 0.005, 0.001]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10)
grid_search.fit(X_train_pca, y_train)

# Get the best SVM model
best_svm_model = grid_search.best_estimator_

# All results
results_df = pd.DataFrame(grid_search.cv_results_)[["param_C", "param_gamma", "mean_test_score"]]

# -----------------------------------------------------------------------------

# Confussion matrix with validation data
filtered_X_val = X_val.drop(columns=dominant_columns)
X_val_scaled = scaler.transform(filtered_X_val)

X_val_scaled_cont = pd.DataFrame(X_val_scaled).drop(columns = binary_columns)
X_val_scaled_bin = pd.DataFrame(X_val_scaled)[binary_columns]

# Run PCA
X_val_pca = pca.transform(X_val_scaled_cont)

# Join dummies back
# Check that there are only 35 columns
X_val_pca.shape
```




    (252, 20)




```python
# Now let's join dummies back
X_val_pca = pd.concat([pd.DataFrame(X_val_pca), X_val_scaled_bin], axis = 1)
X_val_pca.shape
```




    (252, 53)




```python
y_pred = grid_search.best_estimator_.predict(X_val_pca) # Get predictions from the best model
conf_matrix = confusion_matrix(y_val, y_pred)
conf_matrix
```




    array([[ 61,   0],
           [171,  20]])




```python
accuracy = np.mean(y_val == y_pred)
accuracy # 0.32
```




    0.32142857142857145




```python
# Decision scores
decision_scores = grid_search.best_estimator_.decision_function(X_val_pca)

# Set custom threshold
custom_threshold = -0.45  # Adjust as needed

# Make predictions based on custom threshold
y_pred = (decision_scores > custom_threshold).astype(int)
conf_matrix = confusion_matrix(y_val, y_pred)
conf_matrix
```




    array([[ 31,  30],
           [ 20, 171]])




```python
accuracy = np.mean(y_val == y_pred)
accuracy # 0.80

```




    0.8015873015873016




```python

disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["No", "Yes"])
disp.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb292910be0>




    
![png](output_34_1.png)
    



```python
# AUC and ROC
# Calculate false positive rate (fpr) and true positive rate (tpr) for various thresholds
fpr, tpr, thresholds = roc_curve(y_val, decision_scores)

# Calculate the Area Under Curve (AUC)
auc = roc_auc_score(y_val, decision_scores)
auc # 0.82
```




    0.815208994936057




```python
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='skyblue', label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_36_0.png)
    


# ANN model


```python
#%% Artificial Neural Network (ANN)

'''
I have previously identified that there are many highly
correlated variables in the data. Moreover, I showed that by using
PCA and reducing dimensionality, the performance of the SVM model imporves.
Hence, for all future models I will use PCA-reduced data with 20 features.
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow_addons as tfa # for F1 score
```


```python
'''
Many hyperparameters and different models tested. Only the best model is shown here.
In particular, I experimented with layers, activation functions, number of neurons,
and dropout rates. I did not automate this process and tested everything manually.

'''
# Set random 
tf.random.set_seed(124)
np.random.seed(124)
keras.utils.set_random_seed(123)

model = keras.Sequential(
    [
    layers.Dense(35, activation="relu", kernel_initializer="uniform", input_dim = X_train_pca.shape[1]),
    layers.Dropout(0.2),
    layers.Dense(50, activation="relu", kernel_initializer="uniform"),
    layers.Dropout(0.3),
    layers.Dense(35, activation="relu", kernel_initializer="uniform"),
    layers.Dropout(0.2),
    layers.Dense(18, activation="relu", kernel_initializer="uniform"),
    layers.Dropout(0.1),
    layers.Dense(1,  kernel_initializer="uniform", activation = "sigmoid")
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy', tf.keras.metrics.AUC(),
                          keras.metrics.Precision(), keras.metrics.Recall(), tfa.metrics.F1Score(num_classes = 1, threshold = 0.55)])
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_10 (Dense)            (None, 35)                1890      
                                                                     
     dropout_8 (Dropout)         (None, 35)                0         
                                                                     
     dense_11 (Dense)            (None, 50)                1800      
                                                                     
     dropout_9 (Dropout)         (None, 50)                0         
                                                                     
     dense_12 (Dense)            (None, 35)                1785      
                                                                     
     dropout_10 (Dropout)        (None, 35)                0         
                                                                     
     dense_13 (Dense)            (None, 18)                648       
                                                                     
     dropout_11 (Dropout)        (None, 18)                0         
                                                                     
     dense_14 (Dense)            (None, 1)                 19        
                                                                     
    =================================================================
    Total params: 6,142
    Trainable params: 6,142
    Non-trainable params: 0
    _________________________________________________________________



```python
# validation_split: to include 30% of the data for model validation, which prevents overfitting.
results = model.fit(x = X_train_pca, y = y_train, batch_size = 200, epochs = 35, validation_split = 0.20)

```


```python
# Plot results 
plt.plot(results.history['accuracy'], '-o') # 'o' is to show the markers, '-' is to draw the line
plt.plot(results.history['val_accuracy'], '-o')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
```




    <matplotlib.legend.Legend at 0x7fb244708130>




    
![png](output_41_1.png)
    



```python
# Loss
plt.plot(results.history['loss'], '-o')
plt.plot(results.history['val_loss'], '-o')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
```




    <matplotlib.legend.Legend at 0x7fb24478f4f0>




    
![png](output_42_1.png)
    



```python
val_loss, val_acc, val_auc, val_precision, val_recall, f1_score = model.evaluate(X_val_pca, y_val)

val_acc # Accuracy : 0.80
val_auc # AUC : 0.86
val_precision  # Precision : 0.83
val_recall # Recall : 0.92
f1_score # F1 Score : 0.87
```

    8/8 [==============================] - 0s 1ms/step - loss: 0.7651 - accuracy: 0.7976 - auc_2: 0.8625 - precision_2: 0.8153 - recall_2: 0.9476 - f1_score: 0.8765    





    array([0.87651336], dtype=float32)




```python
# Confusion matrix
y_pred = model.predict(X_val_pca)

# Set custom threshold
custom_threshold = 0.35  # Adjust as needed

# Make predictions based on custom threshold
y_pred = (y_pred > custom_threshold).astype(int)
conf_matrix = confusion_matrix(y_val, y_pred)
conf_matrix
```

    8/8 [==============================] - 0s 1ms/step





    array([[ 20,  41],
           [ 10, 181]])




```python
accuracy = np.mean(y_val == y_pred.flatten())
accuracy # 0.80
```




    0.7976190476190477



# LSTM model


```python
#%% LSTM

from tensorflow.keras.layers import LSTM

# Convert to pd dataframe and join
X_train_df = pd.DataFrame(X_train_pca)

# Reset column names to avoid potential problems
column_numbers = list(range(len(X_train_df.columns)))
# Assign numbers to column names
X_train_df = X_train_df.set_axis(column_numbers, axis=1)

# Join Y labels
y_train_df = pd.DataFrame(y_train).reset_index(drop = True)
train_df = pd.concat([X_train_df, y_train_df], axis=1)


X_val_df = pd.DataFrame(X_val_pca)

# Reset column names to avoid potential problems
column_numbers = list(range(len(X_val_df.columns)))
# Assign numbers to column names
X_val_df = X_val_df.set_axis(column_numbers, axis=1)

# Join Y labels
y_val_df = pd.DataFrame(y_val).reset_index(drop = True)
val_df = pd.concat([X_val_df, y_val_df], axis=1)
```


```python
# define function to create windows
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

w2 = WindowGenerator(input_width=10, label_width=1, shift=1,
                     label_columns=['EQY_EM_label'])
w2
```




    Total window size: 11
    Input indices: [0 1 2 3 4 5 6 7 8 9]
    Label indices: [10]
    Label column name(s): ['EQY_EM_label']




```python
# Split data
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# Create window
window = tf.stack([
    np.array(train_df[i:i+w2.total_window_size]) 
    for i in range(len(train_df) - w2.total_window_size + 1)
])

# Splt window on inputs and labels
inputs, labels = w2.split_window(window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {window.shape}')
print(f'Inputs shape: {inputs.shape}')
print(f'Labels shape: {labels.shape}')

```

    All shapes are: (batch, time, features)
    Window shape: (998, 11, 54)
    Inputs shape: (998, 10, 54)
    Labels shape: (998, 1, 1)



```python
# Create tensorflow datasets
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


# Introduce properties
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)


@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val



# Each element is an (inputs, label) pair.
w2.train.element_spec


for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
```

    Inputs shape (batch, time, features): (32, 10, 54)
    Labels shape (batch, time, features): (32, 1, 1)



```python
# Create model
  
'''
Here I tested many many models but one of the simplest
performed better and with greater stability. 

I tested, number of neaurons, layers, activation functions etc.
Did everything manually.
'''
 
# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(100, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1)
# ]) 

from tensorflow.keras import regularizers

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(54, return_sequences=True),  # LSTM layer 1
    # tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.LSTM(27),  # LSTM layer 2 (no need for return_sequences here if final LSTM)
    # tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.Dense(27), #, activation='relu', kernel_initializer="uniform"),  # Dense layer 
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)  # Output layer
])


MAX_EPOCHS = 20

# Define how model is compiled
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


keras.utils.set_random_seed(123)

# Run the model
history = compile_and_fit(lstm_model, w2) 
```

    Epoch 1/20
    32/32 [==============================] - 3s 25ms/step - loss: 0.2849 - mean_absolute_error: 0.4035 - val_loss: 0.4173 - val_mean_absolute_error: 0.5522
    Epoch 2/20
    32/32 [==============================] - 0s 9ms/step - loss: 0.0784 - mean_absolute_error: 0.2085 - val_loss: 0.3948 - val_mean_absolute_error: 0.5510
    Epoch 3/20
    32/32 [==============================] - 0s 9ms/step - loss: 0.0560 - mean_absolute_error: 0.1679 - val_loss: 0.4407 - val_mean_absolute_error: 0.5780
    Epoch 4/20
    32/32 [==============================] - 0s 9ms/step - loss: 0.0471 - mean_absolute_error: 0.1479 - val_loss: 0.4535 - val_mean_absolute_error: 0.5860



```python
import IPython
import IPython.display

IPython.display.clear_output()

# Record loss performance metrics
val_performance = {}
val_performance['LSTM'] = lstm_model.evaluate(w2.val, return_dict=True)

```

    8/8 [==============================] - 0s 3ms/step - loss: 0.4535 - mean_absolute_error: 0.5860



```python
# Now test performance on validation set
val_window = tf.stack([
    np.array(val_df[i:i+w2.total_window_size]) 
    for i in range(len(val_df) - w2.total_window_size + 1)
])

# Input validation data window
val_inputs, val_labels = w2.split_window(val_window)

# Predict based on validation data
predictions = lstm_model(val_inputs)
predictions = predictions[:, 0]
predictions = pd.DataFrame(predictions)

# Get labels
Y_labels = val_labels[:, 0, 0]
Y_labels = pd.DataFrame(Y_labels)


# Set custom threshold
custom_threshold = 0.0  # Adjust as needed

# Make predictions based on custom threshold
predictions = (predictions > custom_threshold).astype(int)
conf_matrix = confusion_matrix(Y_labels, predictions)
conf_matrix
```




    array([[ 36,  25],
           [ 32, 149]])




```python
accuracy = np.mean(Y_labels == predictions)
accuracy # 0.76

```




    0.7644628099173554




```python
# Get more performance metrics

'''
Accuracy : 0.76
Precision : 0.86
Recall : 0.82
AUC : 0.80
'''

from sklearn.metrics import precision_score, recall_score, accuracy_score
accuracy = accuracy_score(Y_labels, predictions)
precision = precision_score(Y_labels, predictions) 
recall = recall_score(Y_labels, predictions) 
```


```python
# Get probability predictions once again for plotting
predictions = lstm_model(val_inputs)
predictions = predictions[:, 0]
predictions = pd.DataFrame(predictions)
Y_labels = val_labels[:, 0, 0]
Y_labels = pd.DataFrame(Y_labels)

# AUC and ROC
# Calculate false positive rate (fpr) and true positive rate (tpr) for various thresholds
fpr, tpr, thresholds = roc_curve(Y_labels, predictions)

# Calculate the Area Under Curve (AUC)
auc = roc_auc_score(Y_labels, predictions)
auc # 0.80
```




    0.8008332578570782




```python
'''
Overall, the model does not look very stable. I managed to find combination
of hyperparameters where model achieved AUC 0.86 and accuracy 0.85, but
by slightly changing inputs, the model performed very differently. 
Eventually I could not replicate such model because I lost specific random seed.

Thus, the best model based on peroformance
is ANN. It also seemed to be more stable.  
'''

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='skyblue', label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](output_57_0.png)
    


# Compile the best model again and measure performance on "test" data


```python
#%% Final model — ANN


X_test, X_train, X_val, y_test, y_train, y_val = split(5)

# Combine train and validation data 
X_train = pd.concat([X_val, X_train], axis = 0)
y_train = pd.concat([y_val, y_train], axis = 0)

# Now let's exclude these columns with value dominating 70%
filtered_X_train = X_train.drop(columns=dominant_columns)

# Center the data first and  scale
scaler = StandardScaler(with_mean=True, with_std=True)
X_centered = scaler.fit_transform(filtered_X_train)

# Now remove dummy columns
X_train_cont = pd.DataFrame(X_centered).drop(columns = binary_columns)
X_train_bin = pd.DataFrame(X_centered)[binary_columns]

# Perform PCA  
# Keep only 20 most valuable PCs
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_cont)

# Now let's join dummies back
X_train_pca = pd.concat([pd.DataFrame(X_train_pca), X_train_bin], axis = 1)

# Check that there are only 53 columns
X_train_pca.shape
```




    (1260, 53)




```python
# Now prepare test data
# Confussion matrix with test data
filtered_X_test = X_test.drop(columns=dominant_columns)
X_test_scaled = scaler.transform(filtered_X_test)

X_test_scaled_cont = pd.DataFrame(X_test_scaled).drop(columns = binary_columns)
X_test_scaled_bin = pd.DataFrame(X_test_scaled)[binary_columns]

# Run PCA
X_test_pca = pca.transform(X_test_scaled_cont)

# Now let's join dummies back
X_test_pca = pd.concat([pd.DataFrame(X_test_pca), X_test_scaled_bin], axis = 1)

# Check that there are only 53 columns
X_test_pca.shape
```




    (65, 53)




```python
# Run the ANN model
# Set random 
tf.random.set_seed(124)
np.random.seed(124)
keras.utils.set_random_seed(123)

model = keras.Sequential(
    [
    layers.Dense(35, activation="relu", kernel_initializer="uniform", input_dim = X_train_pca.shape[1]),
    layers.Dropout(0.2),
    layers.Dense(50, activation="relu", kernel_initializer="uniform"),
    layers.Dropout(0.3),
    layers.Dense(35, activation="relu", kernel_initializer="uniform"),
    layers.Dropout(0.2),
    layers.Dense(18, activation="relu", kernel_initializer="uniform"),
    layers.Dropout(0.1),
    layers.Dense(1,  kernel_initializer="uniform", activation = "sigmoid")
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy', tf.keras.metrics.AUC(),
                          keras.metrics.Precision(), keras.metrics.Recall(), tfa.metrics.F1Score(num_classes = 1, threshold = 0.55)])
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_16 (Dense)            (None, 35)                1890      
                                                                     
     dropout_12 (Dropout)        (None, 35)                0         
                                                                     
     dense_17 (Dense)            (None, 50)                1800      
                                                                     
     dropout_13 (Dropout)        (None, 50)                0         
                                                                     
     dense_18 (Dense)            (None, 35)                1785      
                                                                     
     dropout_14 (Dropout)        (None, 35)                0         
                                                                     
     dense_19 (Dense)            (None, 18)                648       
                                                                     
     dropout_15 (Dropout)        (None, 18)                0         
                                                                     
     dense_20 (Dense)            (None, 1)                 19        
                                                                     
    =================================================================
    Total params: 6,142
    Trainable params: 6,142
    Non-trainable params: 0
    _________________________________________________________________



```python
# validation_split: to include 20% of the data for model validation, which prevents overfitting.
results = model.fit(x = X_train_pca, y = y_train, batch_size = 200, epochs = 35, validation_split = 0.20)

```


```python
# Plot results 
plt.plot(results.history['accuracy'], '-o') # 'o' is to show the markers, '-' is to draw the line
plt.plot(results.history['val_accuracy'], '-o')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
```




    <matplotlib.legend.Legend at 0x7fb23e6b2170>




    
![png](output_63_1.png)
    



```python
# Loss
plt.plot(results.history['loss'], '-o')
plt.plot(results.history['val_loss'], '-o')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
```




    <matplotlib.legend.Legend at 0x7fb247a3b520>




    
![png](output_64_1.png)
    



```python
# Performance on train data
val_loss, val_acc, val_auc, val_precision, val_recall, f1_score = model.evaluate(X_train_pca, y_train)

"""
Performance of train data
Accuracy : 0.8381
AUC : 0.9388
Precision :  0.9012
Recall : 0.7916
F1 Score : 0.84055

"""

val_acc # 0.84
val_auc # 0.94
val_precision  # 0.90
val_recall # 0.79
f1_score # 0.84
```

    40/40 [==============================] - 0s 2ms/step - loss: 0.3762 - accuracy: 0.8381 - auc_3: 0.9393 - precision_3: 0.8998 - recall_3: 0.7931 - f1_score: 0.8397  





    array([0.83965915], dtype=float32)




```python
# Performance on test data
val_loss, val_acc, val_auc, val_precision, val_recall, f1_score = model.evaluate(X_test_pca, y_test)

"""
Performance of test data
Accuracy : 0.8769
AUC : 0.9957
Precision : 0.8095
Recall : 1.0
F1 Score : 0.9067

"""

val_acc # 0.88
val_auc # 0.99
val_precision  # 0.80
val_recall # 1
f1_score # 0.91

```

    3/3 [==============================] - 0s 3ms/step - loss: 0.2968 - accuracy: 0.8923 - auc_3: 0.9957 - precision_3: 0.8293 - recall_3: 1.0000 - f1_score: 0.9189





    array([0.9189189], dtype=float32)




```python
# Confusion matrix

y_pred = model.predict(X_test_pca)

# Set custom threshold
custom_threshold = 0.65  # Adjust as needed

# Make predictions based on custom threshold
y_pred = (y_pred > custom_threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix
```

    3/3 [==============================] - 0s 2ms/step





    array([[26,  5],
           [ 0, 34]])




```python
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["No", "Yes"])
disp.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb247aeceb0>




    
![png](output_68_1.png)
    



```python
'''
Accuracy with adjusted threshold : 0.9231

The model performs very well. The best results I have seen so far. Perhaps, there
is some level of overfitting or the test sample is too small.

Also, I adjusted decision threshold upwards to make predictions more safe.
Overall, the model does not look stagnant and does not miss any profitable opportunities.

Would be interesting to test the model on completely new data and a bigger dataset. 
'''

accuracy = np.mean(y_test == y_pred.flatten())
accuracy # 0.91
```




    0.9230769230769231


