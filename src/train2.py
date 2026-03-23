import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def parse_args():
    parser = argparse.ArgumentParser(description="Train IMDB Sentiment Model")
    parser.add_argument("--c_param", type=float, default=1.0, help="Inverse of regularization strength")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--max_features", type=int, default=5000, help="Top N words to use as features")
    parser.add_argument("--author", type=str, default="Anastasiia", help="Name of the researcher")
    return parser.parse_args()


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path).drop_duplicates()
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df


def vectorize_text(X_train, X_test, max_features):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def train():
    args = parse_args()
    data_path = "C:\\Users\\User\\mlops_lab_1\\data\\row\\dataset.csv"

    df = load_and_preprocess_data(data_path)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )
    X_train, X_test = vectorize_text(X_train_raw, X_test_raw, args.max_features)

    mlflow.sklearn.autolog(log_models=True)

    mlflow.set_experiment("experiment_autolog_100")

    with mlflow.start_run():
        mlflow.set_tag("author", args.author)

        model = LogisticRegression(C=args.c_param, max_iter=args.max_iter)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        print("\n" + "=" * 50)
        print(f"Навчання завершено! Автор: {args.author}")
        print(f"Test Score (Accuracy): {score:.4f}")
        print("Всі метрики, параметри та артефакти залоговано автоматично.")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    train()
