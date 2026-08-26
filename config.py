"""
config.py
---------
Central configuration for the TruthGuard Flask application.

Keeping all configuration in one place makes it easy to switch between
development / testing / production settings without touching app.py.
"""

import os

# Absolute path of the project's root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration shared by every environment."""

    # Secret key used to sign session cookies. In production this MUST be
    # overridden with an environment variable (never hard-code secrets).
    SECRET_KEY = os.environ.get("SECRET_KEY", "truthguard-dev-secret-key-change-me")

    # SQLite database stored inside the instance/ folder.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'truthguard.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Paths to the trained ML artifacts produced by train_model.py
    MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

    # Dataset locations
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    FAKE_CSV = os.path.join(DATASET_DIR, "Fake.csv")
    TRUE_CSV = os.path.join(DATASET_DIR, "True.csv")

    # Upload folder (used for PDF export / temp files)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

    # Max article length accepted by the /predict endpoint (characters)
    MAX_ARTICLE_LENGTH = 20000


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
