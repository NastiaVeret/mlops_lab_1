import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Train IMDB Sentiment Model")
    parser.add_argument("--c_param", type=float, default=1.0, help="Inverse of regularization strength")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--max_features", type=int, default=5000, help="Top N words to use as features")
    parser.add_argument("--author", type=str, default="Student_AI", help="Name of the researcher")
    return parser.parse_args()

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path).drop_duplicates()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def vectorize_text(X_train, X_test, max_features):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf

def log_feature_importance(model, tfidf):
    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]

    feat_importances = pd.DataFrame({'word': feature_names, 'weight': coefficients})
    top_features = pd.concat([
        feat_importances.sort_values(by='weight').head(10),
        feat_importances.sort_values(by='weight').tail(10)
    ])

    plt.figure(figsize=(10, 8))
    sns.barplot(x='weight', y='word', data=top_features, palette='coolwarm', hue='word', legend=False)
    plt.title("Top 10 Positive & Negative Words (Feature Importance)")
    plt.tight_layout()

    plot_path = "feature_importance.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()

def log_confusion_matrix(y_true, y_pred, set_name="test"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f"Confusion Matrix - {set_name.capitalize()}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    plot_path = f"confusion_matrix_{set_name}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()

def train():
    args = parse_args()
    data_path = 'data\\raw\\dataset.csv'

    df = load_and_preprocess_data(data_path)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )

    X_train, X_test, tfidf = vectorize_text(X_train_raw, X_test_raw, args.max_features)

    mlflow.set_experiment("IMDB_Advanced_Analysis")

    with mlflow.start_run():
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("model_type", "LogisticRegression")

        model = LogisticRegression(C=args.c_param, max_iter=args.max_iter)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "train_f1": f1_score(y_train, y_pred_train),
            "test_f1": f1_score(y_test, y_pred_test),
            "train_precision": precision_score(y_train, y_pred_train),
            "test_precision": precision_score(y_test, y_pred_test),
            "train_recall": recall_score(y_train, y_pred_train),
            "test_recall": recall_score(y_test, y_pred_test)
        }

        mlflow.log_params(vars(args))
        mlflow.log_metrics(metrics)

        log_feature_importance(model, tfidf)
        log_confusion_matrix(y_test, y_pred_test, set_name="test")
        log_confusion_matrix(y_train, y_pred_train, set_name="train")

        mlflow.sklearn.log_model(model, "model")

        print("\n" + "="*50)
        print(f"REPORT FOR AUTHOR: {args.author}")
        print("-" * 50)
        print(f"{'Metric':<15} | {'Train':<10} | {'Test':<10}")
        print("-" * 50)
        for m in ["accuracy", "f1", "precision", "recall"]:
            print(f"{m.capitalize():<15} | {metrics[f'train_{m}']:<10.4f} | {metrics[f'test_{m}']:<10.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    train()