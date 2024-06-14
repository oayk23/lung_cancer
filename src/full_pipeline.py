import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .transformers import CustomLabelEncoder,CustomStandardScaler,FeatureSelector
from .constants import CATEGORICAL_VARIABLES,SELECTED_FEATURES


pipe = Pipeline(
    [
        ("CustomLabelEncoder",CustomLabelEncoder(variables=CATEGORICAL_VARIABLES)),
        ("StandardScaler",CustomStandardScaler()),
        ("FeatureSelector",FeatureSelector(selected_features=SELECTED_FEATURES)),
        ("Estimator",LogisticRegression(C=0.0005))
    ]
)




