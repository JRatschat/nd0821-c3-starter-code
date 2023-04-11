# Script to train machine learning model.
from pathlib import Path

import pandas as pd

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    save_as_pickle,
    slice_performance,
    train_model,
)
from sklearn.model_selection import train_test_split

# Add code to load in the data.
data = pd.read_csv(Path().cwd() / "starter/data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function and save encoder and lb.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
save_as_pickle(encoder, Path().cwd() / "starter/model/encoder.sav")
save_as_pickle(lb, Path().cwd() / "starter/model/lb.sav")

# Train and save a model.
model = train_model(X_train, y_train)
save_as_pickle(model, Path().cwd() / "starter/model/model.sav")

# Test model and print evaluation metrics
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(
    f"""
    ##### Model performance #####
    Precision: {round(precision, 4)}
    Recall: {round(recall, 4)}
    Fbeta: {round(fbeta, 4)}
    """
    )

# Calculate and save slicing performance
slice_performance(
    test,
    "salary",
    preds,
    cat_features,
    Path().cwd() / "starter/model/slice_output.txt"
    )
