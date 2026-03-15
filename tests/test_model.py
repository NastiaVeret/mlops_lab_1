import os
import json
import pandas as pd
import pytest

# Constants for paths
DATA_PATH = "data/row/dataset.csv"
TRAIN_DATA_PATH = "data/prepared/train.csv"
TEST_DATA_PATH = "data/prepared/test.csv"
METRICS_PATH = "metrics.json"
MODEL_PATH = "model.pkl"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"


def test_data_validation():
    """
    Перевірка якості та структури вхідних даних (data validation).
    """
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset file not found (maybe ignored by Git/DVC): {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Check if required columns exist
    assert "review" in df.columns, "Column 'review' is missing"
    assert "sentiment" in df.columns, "Column 'sentiment' is missing"

    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

    # Check target classes validity
    valid_sentiments = {"positive", "negative"}
    assert set(df["sentiment"].unique()).issubset(
        valid_sentiments
    ), "Invalid values in 'sentiment' column"

    # Optional: check prepared validation datasets if exist
    if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        assert len(train_df) > 0, "Train dataset is empty"
        assert len(test_df) > 0, "Test dataset is empty"


def test_config_validation():
    """
    Перевірка валідності конфігурації (Hydra).
    """
    # Assuming config/config.yaml is the main file
    config_path = "config/config.yaml"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"

    # Optional: load yaml to ensure it is valid
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert config is not None, "Configuration file is empty or invalid"
    assert "mlflow" in config, "Missing 'mlflow' section in configuration"
    assert "seed" in config, "Missing 'seed' value in configuration"


def test_artifacts_creation():
    """
    Перевірка коректності створення артефактів (model.pkl, metrics.json, confusion_matrix.png).
    """
    artifacts = [MODEL_PATH, METRICS_PATH, CONFUSION_MATRIX_PATH]
    for artifact in artifacts:
        assert os.path.exists(artifact), f"Artifact '{artifact}' was not generated"
        assert os.path.getsize(artifact) > 0, f"Artifact '{artifact}' is empty"

    # Check if we can properly load the model
    import joblib

    try:
        model = joblib.load(MODEL_PATH)
        assert model is not None, "Model loaded from 'model.pkl' is None"
    except Exception as e:
        pytest.fail(f"Could not load 'model.pkl': {e}")


def test_quality_gate():
    """
    Проходження Quality Gate за метриками (наприклад, f1 >= threshold).
    """
    assert os.path.exists(METRICS_PATH), f"Metrics file not found: {METRICS_PATH}"

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert "test_f1" in metrics, "Metric 'test_f1' not found in metrics.json"

    # Threshold definition
    MIN_F1_SCORE = 0.70
    f1_score = metrics["test_f1"]

    assert f1_score >= MIN_F1_SCORE, (
        f"Quality Gate failed: F1 score ({f1_score:.4f}) "
        f"is below the minimum threshold ({MIN_F1_SCORE})"
    )
