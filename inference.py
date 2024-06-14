import os
import pandas as pd
import joblib




def main():
    model_path = r"C:\Users\omera\Desktop\lung_cancer\models\trained_pipeline.pkl"

    pipeline = joblib.load(model_path)

    data_path = r"C:\Users\omera\Desktop\lung_cancer\data\x_test.csv"

    data = pd.read_csv(data_path)

    prediction = pipeline.predict(data)

    print(prediction)



main()