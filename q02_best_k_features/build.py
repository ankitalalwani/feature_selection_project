# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    X_new = SelectPercentile(f_regression, k).fit_transform(X, y)
    return X_new.tolist()

percentile_k_features(data)





