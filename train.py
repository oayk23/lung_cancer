import pandas as pd
import os
from sklearn.metrics import accuracy_score
import joblib
import sys

from src.full_pipeline import pipe

sys.path.append("..")

def main():
    print("------------------Training Started------------------------")
    data_root = r"C:\Users\omera\Desktop\lung_cancer\data"
    
    x_train = pd.read_csv(os.path.join(data_root,"x_train.csv"))
    y_train = pd.read_csv(os.path.join(data_root,"y_train.csv"))
    x_test = pd.read_csv(os.path.join(data_root,"x_test.csv"))
    y_test = pd.read_csv(os.path.join(data_root,"y_test.csv"))
    
    print("Fitting the pipeline...")
    pipe.fit(x_train,y_train)

    print("Predicting on x_test...")
    y_preds = pipe.predict(x_test)
    acc = accuracy_score(y_test,y_preds)

    print("Pipeline model gathered accuracy:",acc)

    model_path = r"C:\Users\omera\Desktop\lung_cancer\models"
    pipeline_path = os.path.join(model_path,"trained_pipeline.pkl")
    print(f"Saving Pipeline to {model_path}")
    joblib.dump(pipe,pipeline_path)

    print(f"Pipeline saved to {pipeline_path}")

    print("---------------------Training Ended------------------------")



main()