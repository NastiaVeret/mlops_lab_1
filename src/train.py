"""
Навчання моделей з повним звітом: HPO на валідації, метрики, confusion matrix,
архітектура pipeline, криві навчання та графіки валідації (CV / learning curve).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    PredefinedSplit,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--author", type=str, default="Anastasiia")
    parser.add_argument(
        "--models",
        type=str,
        default="logistic_regression",
        help="Через кому: logistic_regression, linearsvc, multinomial_nb",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Частка train для валідації під час підбору гіперпараметрів",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--n_iter_search",
        type=int,
        default=25,
        help="Кількість випадкових комбінацій для RandomizedSearchCV",
    )
    parser.add_argument(
        "--learning_curve_cv",
        type=int,
        default=3,
        help="Кількість фолдів для learning_curve (на повному train після HPO)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Паралелізм sklearn (-1 = усі ядра; у обмежених середовищах вкажіть 1)",
    )
    return parser.parse_args()


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def make_pipelines(random_state: int, n_jobs_clf: int) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(stop_words="english", sublinear_tf=True),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=random_state,
                        n_jobs=n_jobs_clf,
                        solver="saga",
                    ),
                ),
            ]
        ),
        "linearsvc": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(stop_words="english", sublinear_tf=True),
                ),
                (
                    "clf",
                    LinearSVC(random_state=random_state, max_iter=8000),
                ),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(stop_words="english", sublinear_tf=True),
                ),
                ("clf", MultinomialNB()),
            ]
        ),
    }


def param_distributions_for(model_key: str) -> Dict[str, Any]:
    from scipy.stats import loguniform, randint, uniform

    common_tfidf = {
        "tfidf__max_features": randint(2000, 12001),
        "tfidf__min_df": randint(1, 4),
        "tfidf__max_df": uniform(0.85, 0.14),
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    }
    if model_key == "logistic_regression":
        return {
            **common_tfidf,
            "clf__C": loguniform(1e-2, 1e2),
            "clf__penalty": ["l2", "l1"],
        }
    if model_key == "linearsvc":
        return {
            **common_tfidf,
            "clf__C": loguniform(1e-2, 1e2),
            "clf__loss": ["squared_hinge"],
        }
    if model_key == "multinomial_nb":
        return {
            **common_tfidf,
            "clf__alpha": loguniform(1e-3, 1e1),
        }
    raise ValueError(f"Невідома модель: {model_key}")


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    if y.dtype == object or hasattr(y.iloc[0] if hasattr(y, "iloc") else y[0], "lower"):
        s = pd.Series(y)
        mapped = s.map({"positive": 1, "negative": 0})
        if mapped.isna().any():
            raise ValueError("Очікувані sentiment: positive/negative або 0/1")
        return mapped.to_numpy()
    return np.asarray(y)


def architecture_report(pipe: Pipeline) -> str:
    buf = io.StringIO()
    print("=== Pipeline (архітектура) ===", file=buf)
    print(pipe, file=buf)
    print("\n=== Кроки та типи ===", file=buf)
    for name, step in pipe.steps:
        print(f"  [{name}] {type(step).__name__}", file=buf)
    print("\n=== Параметри (get_params) ===", file=buf)
    for k, v in sorted(pipe.get_params(deep=False).items()):
        print(f"  {k}: {v}", file=buf)
    deep = pipe.get_params(deep=True)
    print("\n=== Усі ключові параметри (deep, скорочено) ===", file=buf)
    for k in sorted(deep.keys()):
        if "__" in k and not k.startswith("memory"):
            print(f"  {k}: {deep[k]}", file=buf)
    return buf.getvalue()


def score_for_roc(pipe: Pipeline, X: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1], None
    clf = pipe.named_steps["clf"]
    Xt = pipe.named_steps["tfidf"].transform(X)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(Xt)
        return None, np.ravel(scores)
    return None, None


def collect_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    y_decision: Optional[np.ndarray],
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_binary": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall_binary": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    m["specificity"] = specificity
    m["sensitivity"] = sensitivity

    scores_for_roc = y_proba if y_proba is not None else y_decision
    if scores_for_roc is not None:
        m["roc_auc"] = float(roc_auc_score(y_true, scores_for_roc))
        m["average_precision"] = float(average_precision_score(y_true, scores_for_roc))
    if y_proba is not None:
        try:
            proba_full = np.column_stack([1 - y_proba, y_proba])
            m["log_loss"] = float(log_loss(y_true, proba_full, labels=[0, 1]))
        except Exception:
            pass
    return m


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, prefix: str, labels: List[str]) -> List[str]:
    paths = []
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.title("Confusion matrix (counts)")
    p1 = f"{prefix}confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=120)
    plt.close()
    paths.append(p1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax2, cmap="Blues", values_format=".3f", normalize="true")
    plt.title("Confusion matrix (normalized)")
    p2 = f"{prefix}confusion_matrix_normalized.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=120)
    plt.close()
    paths.append(p2)
    return paths


def plot_hyperparameter_search(search: RandomizedSearchCV, prefix: str) -> List[str]:
    cv = search.cv_results_
    means = np.asarray(cv["mean_test_score"])
    stds = np.asarray(cv["std_test_score"])
    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, means, "o-", label="mean val score")
    ax.fill_between(x, means - stds, means + stds, alpha=0.25)
    ax.set_xlabel("Iteration (RandomizedSearchCV)")
    ax.set_ylabel("Score on validation fold")
    ax.set_title("Підбір гіперпараметрів: валідаційний score по ітераціях")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"{prefix}validation_search_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cumulative_best = np.maximum.accumulate(means)
    ax2.plot(range(len(cumulative_best)), cumulative_best, "-", color="C1")
    ax2.set_xlabel("Trial index (order of RandomizedSearchCV)")
    ax2.set_ylabel("Best validation score so far")
    ax2.set_title("Накопичувальний найкращий валідаційний score")
    ax2.grid(True, alpha=0.3)
    path2 = f"{prefix}validation_best_so_far.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=120)
    plt.close()
    return [path, path2]


def plot_learning_curve_chart(
    pipe: Pipeline,
    X: Any,
    y: np.ndarray,
    cv_splits: int,
    random_state: int,
    prefix: str,
    n_jobs: int,
) -> str:
    train_sizes, train_scores, val_scores = learning_curve(
        pipe,
        X,
        y,
        cv=cv_splits,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=n_jobs,
        shuffle=True,
        random_state=random_state,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Train F1")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, val_mean, "o-", label="CV validation F1")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 score")
    ax.set_title("Learning curve (train vs cross-val)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"{prefix}learning_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_roc_pr(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    y_decision: Optional[np.ndarray],
    prefix: str,
) -> List[str]:
    paths: List[str] = []
    scores = y_proba if y_proba is not None else y_decision
    if scores is None:
        return paths
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, scores):.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve (test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = f"{prefix}roc_curve.png"
    plt.tight_layout()
    plt.savefig(p, dpi=120)
    plt.close()
    paths.append(p)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision–Recall curve (test)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    p2 = f"{prefix}pr_curve.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=120)
    plt.close()
    paths.append(p2)
    return paths


def train_one_model(
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    pipe_factories: Dict[str, Pipeline],
    primary: bool,
) -> None:
    prefix = "" if primary else f"{model_key}_"
    X_train = train_df["review"]
    y_train = _ensure_binary_labels(train_df["sentiment"].values)
    X_test = test_df["review"]
    y_test = _ensure_binary_labels(test_df["sentiment"].values)

    from sklearn.model_selection import train_test_split

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train,
    )
    X_combined = pd.concat([X_sub, X_val], axis=0)
    y_combined = np.concatenate([y_sub, y_val])
    test_fold = np.concatenate([np.full(len(X_sub), -1, dtype=int), np.zeros(len(X_val), dtype=int)])
    cv_split = PredefinedSplit(test_fold=test_fold)

    base_pipe = pipe_factories[model_key]
    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_distributions_for(model_key),
        n_iter=args.n_iter_search,
        scoring="f1",
        cv=cv_split,
        refit=True,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=1,
    )
    print(f"\n{'=' * 60}\nМодель: {model_key}\nПідбір гіперпараметрів (валідація…)\n")
    search.fit(X_combined, y_combined)
    best: Pipeline = search.best_estimator_

    arch_text = architecture_report(best)
    arch_path = f"{prefix}architecture.txt"
    Path(arch_path).write_text(arch_text, encoding="utf-8")
    print(arch_text)

    y_pred_test = best.predict(X_test)
    y_proba, y_decision = score_for_roc(best, X_test)
    metrics_test = collect_metrics(y_test, y_pred_test, y_proba, y_decision)

    y_pred_train = best.predict(X_train)
    y_proba_tr, y_decision_tr = score_for_roc(best, X_train)
    metrics_train = collect_metrics(y_train, y_pred_train, y_proba_tr, y_decision_tr)

    report_txt = classification_report(y_test, y_pred_test, target_names=["negative", "positive"])
    report_path = f"{prefix}classification_report.txt"
    Path(report_path).write_text(report_txt, encoding="utf-8")
    print(report_txt)

    combined_metrics = {
        "model": model_key,
        "best_cv_score_f1": float(search.best_score_),
        "best_params": search.best_params_,
        "test": metrics_test,
        "train_eval": metrics_train,
    }
    flat_for_mlflow: Dict[str, float] = {
        "cv_best_f1": float(search.best_score_),
    }
    for k, v in metrics_test.items():
        flat_for_mlflow[f"test_{k}"] = float(v)
    for k, v in metrics_train.items():
        flat_for_mlflow[f"train_{k}"] = float(v)

    metrics_json_path = f"{prefix}metrics_full.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, ensure_ascii=False, indent=2)

    qc_metrics = {
        "accuracy": metrics_test["accuracy"],
        "f1": metrics_test["f1_binary"],
        "test_accuracy": metrics_test["accuracy"],
        "test_f1": metrics_test["f1_binary"],
        "test_precision": metrics_test["precision_binary"],
        "test_recall": metrics_test["recall_binary"],
    }
    if primary:
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(qc_metrics, f, ensure_ascii=False, indent=2)

    cm_paths = plot_confusion_matrices(y_test, y_pred_test, prefix, ["neg", "pos"])
    if primary:
        import shutil

        shutil.copy2(cm_paths[0], "confusion_matrix.png")

    search_plots = plot_hyperparameter_search(search, prefix)
    lc_path = plot_learning_curve_chart(
        best,
        X_train,
        y_train,
        args.learning_curve_cv,
        args.random_state,
        prefix,
        args.n_jobs,
    )
    roc_paths = plot_roc_pr(y_test, y_proba, y_decision, prefix)

    cv_dump = {
        "params": search.cv_results_["params"],
        "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
        "std_test_score": search.cv_results_["std_test_score"].tolist(),
    }
    cv_path = f"{prefix}cv_results.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_dump, f, indent=2)

    if primary:
        import joblib

        joblib.dump(best, "model.pkl")

    artifact_paths = [arch_path, metrics_json_path, report_path, cv_path, lc_path] + search_plots + cm_paths + roc_paths
    for p in artifact_paths:
        if p and Path(p).exists():
            mlflow.log_artifact(p)

    mlflow.log_metrics(flat_for_mlflow)
    mlflow.log_params({f"best__{k}": str(v) for k, v in search.best_params_.items()})
    mlflow.sklearn.log_model(best, artifact_path=f"model_{model_key}")

    roc_disp = metrics_test.get("roc_auc")
    roc_str = f"{roc_disp:.4f}" if roc_disp is not None else "n/a"
    print(
        f"\nНайкращі параметри (за F1 на валідації): {search.best_params_}\n"
        f"Найкращий CV F1: {search.best_score_:.4f}\n"
        f"Тест accuracy={metrics_test['accuracy']:.4f}, "
        f"F1={metrics_test['f1_binary']:.4f}, "
        f"ROC-AUC={roc_str}\n"
    )


def train(args=None):
    if args is None:
        args = parse_args()

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    model_keys = [_slug(m) for m in args.models.split(",") if m.strip()]
    factories = make_pipelines(args.random_state, args.n_jobs)
    for k in model_keys:
        if k not in factories:
            print(f"Пропуск невідомої моделі: {k}", file=sys.stderr)
            sys.exit(1)

    mlflow.set_experiment("sentiment_analysis_stages")

    with mlflow.start_run():
        mlflow.set_tag("author", args.author)
        mlflow.log_params(
            {
                "models": args.models,
                "val_size": args.val_size,
                "n_iter_search": args.n_iter_search,
                "learning_curve_cv": args.learning_curve_cv,
                "random_state": args.random_state,
                "n_jobs": args.n_jobs,
            }
        )

        for i, model_key in enumerate(model_keys):
            is_primary = i == 0
            run_name = f"{model_key}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tag("model", model_key)
                train_one_model(
                    model_key,
                    train_df,
                    test_df,
                    args,
                    factories,
                    primary=is_primary,
                )


if __name__ == "__main__":
    train(None)
