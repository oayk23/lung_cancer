import pandas as pd
import os
from sklearn.model_selection import train_test_split



def main():

    data_root = r"C:\Users\omera\Desktop\lung_cancer\data"
    data_name = "lung_cancer_data.csv"
    lung_cancer_data_path = os.path.join(data_root,data_name)

    data = pd.read_csv(lung_cancer_data_path)
    if not isinstance(data,pd.DataFrame):
        raise ValueError("Data is not a Dataframe.")

    target = "Stage"
    id = "Patient_ID"


    x_train,x_test,y_train,y_test = train_test_split(
        data.drop([target,id],axis=1),
        data[target],
        test_size=0.2,
        random_state=0
    )

    x_train.to_csv(os.path.join(data_root,"x_train.csv"),index=False)
    x_test.to_csv(os.path.join(data_root,"x_test.csv"),index=False)
    y_train.to_csv(os.path.join(data_root,"y_train.csv"),index=False)
    y_test.to_csv(os.path.join(data_root,"y_test.csv"),index=False)



main()

