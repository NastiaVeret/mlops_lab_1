import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os
import time
import numpy as np


def load_and_vectorize(train_path, test_path, max_features=5000):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Data files not found. Run prepare script first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = tfidf.fit_transform(train_df["review"])
    X_test = tfidf.transform(test_df["review"])
    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    return X_train, X_test, y_train, y_test


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(f"{cfg.mlflow.experiment_name}_comparison")

    train_path = "data/prepared/train.csv"
    test_path = "data/prepared/test.csv"
    X_train, X_test, y_train, y_test = load_and_vectorize(train_path, test_path)

    results = {}

    def objective(trial, X, y, cfg):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=cfg.seed
        )
        return cross_val_score(model, X, y, cv=3, scoring="f1").mean()

    samplers = {
        "Random": optuna.samplers.RandomSampler(seed=cfg.seed),
        "TPE": optuna.samplers.TPESampler(seed=cfg.seed),
    }

    plt.figure(figsize=(10, 6))

    for name, sampler in samplers.items():
        print(f"\nRunning HPO with {name} Sampler...")
        start_time = time.time()

        study = optuna.create_study(direction="maximize", sampler=sampler)

        with mlflow.start_run(run_name=f"Comparison_{name}"):
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, cfg),
                n_trials=cfg.hpo.n_trials,
            )

            duration = time.time() - start_time
            values = [t.value for t in study.trials if t.value is not None]
            best_values = np.maximum.accumulate(values)

            results[name] = {
                "best_value": study.best_value,
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "duration": duration,
                "best_params": study.best_params,
            }

            mlflow.log_params(
                {"sampler": name, "n_trials": cfg.hpo.n_trials, "duration": duration}
            )
            mlflow.log_metrics(
                {
                    "best_f1": study.best_value,
                    "mean_f1": np.mean(values),
                    "median_f1": np.median(values),
                    "std_f1": np.std(values),
                }
            )

            plt.plot(
                range(1, len(best_values) + 1),
                best_values,
                label=f"{name} (Best: {study.best_value:.4f})",
            )

    plt.title("Sampler Comparison: Best-so-far F1 Score")
    plt.xlabel("Trial")
    plt.ylabel("Best F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("sampler_comparison.png")

    with mlflow.start_run(run_name="Sampler_Comparison_Summary"):
        mlflow.log_artifact("sampler_comparison.png")
        print("\n=== Comparison Results ===")
        for name, res in results.items():
            print(f"\nSampler: {name}")
            print(f"  Best Value: {res['best_value']:.4f}")
            print(f"  Mean Value: {res['mean']:.4f}")
            print(f"  Median Value: {res['median']:.4f}")
            print(f"  Std Dev: {res['std']:.4f}")
            print(f"  Duration: {res['duration']:.2f}s")
            print(f"  Best Params: {res['best_params']}")

        print("\nSummary plot saved as 'sampler_comparison.png' and logged to MLflow.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
