def add_features(X):
    X = X.copy()

    # avoid division by zero
    X["AvgCharges"] = X["TotalCharges"] / (X["tenure"] + 1)

    return X