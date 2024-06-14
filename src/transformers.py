import pandas as pd

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin


class CustomStandardScaler(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.sc = StandardScaler()

    def fit(self,x,y=None):
        return self
    
    def transform(self,x,y=None):
        return pd.DataFrame(self.sc.fit_transform(x),columns=x.columns)
    

class FeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self,selected_features):
        self.selected_features = selected_features

    def fit(self,x,y):
        return self
    
    def transform(self,x,y=None):

        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        missing_cols = [col for col in self.selected_features if col not in x.columns]
        if missing_cols:
            raise ValueError(f"These columns are not in the DataFrame: {missing_cols}")
        
        return x[self.selected_features]


class CustomLabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,variables):
        self.variables = variables
        self.le = LabelEncoder()

    def fit(self,x,y):
        return self
    def transform(self,x):
        x = x.copy()

        for variable in self.variables:
            x[variable] = self.le.fit_transform(x[variable])
        return x