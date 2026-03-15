import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import mlflow
import mlflow.sklearn
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--c_param", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--author", type=str, default="Anastasiia")
    return parser.parse_args()


def log_metrics_and_plots(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
    }
    mlflow.log_metrics(metrics)

    import json

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact("confusion_matrix.png")


def train():
    args = parse_args()

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    tfidf = TfidfVectorizer(max_features=args.max_features, stop_words="english")
    X_train = tfidf.fit_transform(train_df["review"])
    X_test = tfidf.transform(test_df["review"])
    y_train, y_test = train_df["sentiment"], test_df["sentiment"]

    print("Test")

    mlflow.set_experiment("sentiment_analysis_stages")

    with mlflow.start_run():
        mlflow.set_tag("author", args.author)
        mlflow.log_params(vars(args))

        model = LogisticRegression(C=args.c_param, max_iter=args.max_iter)
        model.fit(X_train, y_train)

        log_metrics_and_plots(model, X_train, y_train, X_test, y_test)
        mlflow.sklearn.log_model(model, "model")

        import joblib

        joblib.dump(model, "model.pkl")


if __name__ == "__main__":
    train()
