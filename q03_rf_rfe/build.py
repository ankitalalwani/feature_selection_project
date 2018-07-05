# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(df):
    
    # create a base classifier used to evaluate a subset of attributes
    model = RandomForestClassifier()

    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 17)
    rfe = rfe.fit(X, y)

    # summarize the selection of the attributes
    top_features = X.iloc[0,rfe.ranking_ == 1].index.tolist()
    return top_features

rf_rfe(data)




