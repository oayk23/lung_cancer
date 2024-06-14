import os
import pandas as pd

NUMERICAL_VARIABLES = ['Age', 'Tumor_Size_mm', 'Survival_Months', 'Performance_Status', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Blood_Pressure_Pulse', 'Hemoglobin_Level', 'White_Blood_Cell_Count', 'Platelet_Count', 'Albumin_Level', 'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level', 'Aspartate_Aminotransferase_Level', 'Creatinine_Level', 'LDH_Level', 'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level', 'Potassium_Level', 'Sodium_Level', 'Smoking_Pack_Years']
CATEGORICAL_VARIABLES = ['Gender', 'Smoking_History', 'Tumor_Location', 'Treatment', 'Ethnicity', 'Insurance_Type', 'Family_History', 'Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease', 'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease', 'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other']
SELECTED_FEATURES = ['Age','Smoking_History','Tumor_Size_mm','Tumor_Location','Insurance_Type','Comorbidity_Autoimmune_Disease','Comorbidity_Other','Blood_Pressure_Systolic','Blood_Pressure_Diastolic','White_Blood_Cell_Count','Platelet_Count','Albumin_Level','Calcium_Level','Glucose_Level','Potassium_Level','Sodium_Level','Smoking_Pack_Years']