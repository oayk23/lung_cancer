import pandas as pd
import os
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler


def main():

    data_root = r"C:\Users\omera\Desktop\lung_cancer\data"
    data_name = "lung_cancer_data.csv"
    lung_cancer_data_path = os.path.join(data_root,data_name)

    data = pd.read_csv(lung_cancer_data_path)
    if not isinstance(data,pd.DataFrame):
        raise ValueError("Data is not a Dataframe.")

    target = "Stage"
    id = "Patient_ID"

    CATEGORICAL_VARIABLES = [var for var in data.columns if data[var].dtype == "O" and var != target and var != id]

    x = data.drop([target,id],axis=1)
    y = data[target]

    le = LabelEncoder()

    for var in CATEGORICAL_VARIABLES:
        x[var] = le.fit_transform(x[var])
    
    sc = StandardScaler()

    x = pd.DataFrame(sc.fit_transform(x,y),columns=x.columns)
    
    selector = SelectFromModel(LogisticRegression(C=0.0005))

    selector.fit(x,y)

    selected_features = x.columns[(selector.get_support())]

    pd.Series(selected_features).to_csv(os.path.join(data_root,'selected_features.csv'), index=False)

main()


