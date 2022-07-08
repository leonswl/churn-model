# perform extraction, transformation and loading of dataset
# extraction: dataset obtained from kaggle hence extraction is minimal
# transformation: dataset is mostly clean hence minimal transformation is required
# 


import pandas as pd
import numpy as np

def etl():
    # read dataset
    df = pd.read_csv('data/Telco-Customer-Churn.csv')

    # Drop irrelevant attributes
    df1 = df.drop(['customerID'],axis=1)

    # Convert empty strings to none values
    df1['TotalCharges'] = df1['TotalCharges'].replace(" ",None).dropna().apply(lambda x:float(x))

    # drop all NaN values
    df_dropna = df1.dropna()

    df_dropna.to_parquet('data/preprocessed.parquet')

    print(f"""
    etl.py executed successfully.
    """)

if __name__ == "__main__":
    etl()