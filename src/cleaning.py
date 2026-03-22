import pandas as pd

def clean_data(df):
    df = df.copy()

    # Fix datatype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop irrelevant
    df = df.drop("customerID", axis=1)

    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df