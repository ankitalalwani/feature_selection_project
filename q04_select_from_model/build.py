# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):

    # create a base classifier used to evaluate a subset of attributes
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    
    model = RandomForestClassifier(random_state=9)
    model.fit(X, y)
    
    sfm = SelectFromModel(model, prefit=True)
   

    

    return X.iloc[0,sfm.get_support()].index.tolist()

select_from_model(data)




