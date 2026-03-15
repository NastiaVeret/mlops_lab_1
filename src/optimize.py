import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_validate
import os
import hashlib
import subprocess


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def get_file_hash(path: str) -> str:
    """Calculate MD5 hash of a file."""
    if not os.path.exists(path):
        return "not_found"
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_and_vectorize(train_path, test_path, max_features=5000):
    """Load data and apply TF-IDF vectorization."""
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
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    train_path = "data/prepared/train.csv"
    test_path = "data/prepared/test.csv"

    X_train, X_test, y_train, y_test = load_and_vectorize(train_path, test_path)

    def objective(trial):
        """Optuna objective function with nested MLflow runs."""
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            model_name = cfg.model.name

            if "Random Forest" in model_name:
                n_estimators = trial.suggest_int(
                    "n_estimators",
                    cfg.hpo.search_space["model.n_estimators"].low,
                    cfg.hpo.search_space["model.n_estimators"].high,
                )
                max_depth = trial.suggest_int(
                    "max_depth",
                    cfg.hpo.search_space["model.max_depth"].low,
                    cfg.hpo.search_space["model.max_depth"].high,
                )

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=cfg.seed,
                )
                params = {"n_estimators": n_estimators, "max_depth": max_depth}

            elif "Logistic Regression" in model_name:
                C = trial.suggest_float("C", 0.01, 10.0, log=True)
                model = LogisticRegression(
                    C=C, max_iter=cfg.model.max_iter, random_state=cfg.seed
                )
                params = {"C": C}
            else:
                raise ValueError(f"Model type '{model_name}' is not supported.")

            scoring = ["accuracy", "precision", "recall", "f1"]
            cv_results = cross_validate(model, X_train, y_train, cv=3, scoring=scoring)

            for metric in scoring:
                mlflow.log_metric(f"cv_{metric}", cv_results[f"test_{metric}"].mean())

            primary_metric = cfg.hpo.metric
            score = cv_results[f"test_{primary_metric}"].mean()

            mlflow.log_params(params)
            mlflow.log_metric(primary_metric, score)
            mlflow.set_tags(
                {
                    "trial_number": trial.number,
                    "sampler": cfg.hpo.sampler,
                    "model_type": model_name,
                    "seed": cfg.seed,
                }
            )

            return score

    with mlflow.start_run(run_name=f"HPO_{cfg.model.name}_Study"):
        if cfg.hpo.sampler == "TPE":
            sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        else:
            sampler = optuna.samplers.RandomSampler(seed=cfg.seed)

        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)

        with open("config_copy.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact("config_copy.yaml", artifact_path="config")
        os.remove("config_copy.yaml")

        mlflow.log_params(
            {
                "model_type": cfg.model.name,
                "sampler": cfg.hpo.sampler,
                "n_trials": cfg.hpo.n_trials,
                "metric": cfg.hpo.metric,
                "seed": cfg.seed,
            }
        )

        data_train_hash = get_file_hash(train_path)
        data_test_hash = get_file_hash(test_path)
        git_hash = get_git_revision_hash()

        mlflow.set_tags(
            {
                "git_commit": git_hash,
                "data_train_hash": data_train_hash,
                "data_test_hash": data_test_hash,
                "reproducible": "true",
            }
        )

        print(f"Starting Study optimization for {cfg.model.name}...")
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        mlflow.set_tag("best_trial_number", study.best_trial.number)
        mlflow.log_params(study.best_params)
        mlflow.log_metric(f"best_{cfg.hpo.metric}", study.best_value)

        print("\nTraining final model with best parameters...")
        if "Random Forest" in cfg.model.name:
            final_model = RandomForestClassifier(
                **study.best_params, random_state=cfg.seed
            )
        elif "Logistic Regression" in cfg.model.name:
            final_model = LogisticRegression(**study.best_params, random_state=cfg.seed)
        else:
            final_model = None

        if final_model is not None:
            final_model.fit(X_train, y_train)

            y_pred = final_model.predict(X_test)
            test_metrics = {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, average="binary"),
                "test_recall": recall_score(y_test, y_pred, average="binary"),
                "test_f1": f1_score(y_test, y_pred, average="binary"),
            }
            mlflow.log_metrics(test_metrics)
            print(f"Final Model Test Metrics: {test_metrics}")

            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="best_model",
                registered_model_name=f"{cfg.model.name.replace(' ', '_')}_Best",
            )

            try:
                client = mlflow.tracking.MlflowClient()
                mv = client.get_latest_versions(
                    f"{cfg.model.name.replace(' ', '_')}_Best", stages=["None"]
                )[0]
                client.transition_model_version_stage(
                    name=mv.name, version=mv.version, stage="Staging"
                )
                print(f"Model version {mv.version} transitioned to Staging.")
            except Exception as e:
                print(f"Model registration error: {e}")

        print("\n" + "=" * 40)
        print(f"OPTUNA STUDY COMPLETED")
        print(f"Best Parameters: {study.best_params}")
        print(f"Best {cfg.hpo.metric}: {study.best_value:.4f}")
        print("=" * 40)


if __name__ == "__main__":
    main()
