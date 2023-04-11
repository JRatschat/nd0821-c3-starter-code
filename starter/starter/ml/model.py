import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def save_as_pickle(object, path):
    """ Saves object to pickle.

    Inputs
    ------
    object: ???
        Object to be saved.
    path: str
        Directory for saving pickle file.
    Returns
    -------
    None
    """
    pickle.dump(object, open(path, "wb"))


def slice_performance(data, label, preds, cols, path):
    """ Calculates sliced model performance and saves output to path.

    Inputs
    ------
    data: pd.DataFrame
        Data frame with input data.
    label: str
        Name of column that contains label.
    preds: pd.Series
        Model's predictions.
    cols: List[str]
        List of column names that should be sliced (categorical values)
    path: str
        Path to which slice performance should be written to.
    Returns
    -------
    None
    """
    data["preds"] = preds
    label_category_0 = data[label].unique()[0]
    data["label_true"] = [0 if x==label_category_0 else 1 for x in data[label]]
    with open(path, "w") as f:
        print("Column", "Category", "Precision", "Recall", "Fbeta", file=f)
        for col in cols:
            for category in data[col].unique():
                label_true = data[lambda x: x[col] == category]["label_true"]
                preds = data[lambda x: x[col] == category]["preds"]
                precision, recall, fbeta = compute_model_metrics(label_true, preds)
                print(col, category, round(precision, 4), round(recall, 4), round(fbeta, 4), file=f)
