import pandas as pd
import numpy as np

def etl():
    # read dataset
    df = pd.read_csv('data/Telco-Customer-Churn.csv')

    # Drop irrelevant attributes
    df1 = df.drop(['customerID'],axis=1)

    # Convert empty strings to none values
    df1['TotalCharges'] = df1['TotalCharges'].replace(" ",None).dropna().apply(lambda x:float(x))

    df1.to_parquet('data/preprocessed.parquet')

if __name__ == "__main__":
    etl()