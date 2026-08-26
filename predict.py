"""
predict.py
----------
Loads the trained model + vectorizer once, and exposes a simple
`predict_news(text)` function used by the Flask routes and the REST API.
"""

import os
import joblib
import numpy as np

from config import Config
from utils.text_preprocessing import clean_text, get_suspicious_words


class NewsPredictor:
    """Wraps the trained model + vectorizer behind a clean interface."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.ready = False
        self._load()

    def _load(self):
        if os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.VECTORIZER_PATH):
            self.model = joblib.load(Config.MODEL_PATH)
            self.vectorizer = joblib.load(Config.VECTORIZER_PATH)
            self.ready = True
        else:
            self.ready = False

    def predict(self, raw_text: str) -> dict:
        """
        Predict whether `raw_text` is Fake or Real news.

        Returns a dict with:
            label: "Fake" | "Real"
            confidence: float 0-100
            suspicious_words: list[str]
        """
        if not self.ready:
            raise RuntimeError(
                "No trained model found. Run 'python train_model.py' first "
                "(after generating or downloading a dataset)."
            )

        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("Article text became empty after cleaning. Please provide more text.")

        vec = self.vectorizer.transform([cleaned])
        pred_label = int(self.model.predict(vec)[0])  # 0 = Fake, 1 = Real

        # Confidence: use predict_proba if available, otherwise decision_function
        confidence = self._compute_confidence(vec, pred_label)

        suspicious = []
        if pred_label == 0:  # only worth highlighting suspicious words for Fake predictions
            try:
                suspicious = get_suspicious_words(raw_text, self.vectorizer, self.model)
            except Exception:
                suspicious = []

        return {
            "label": "Fake" if pred_label == 0 else "Real",
            "label_code": pred_label,
            "confidence": round(confidence, 2),
            "suspicious_words": suspicious,
        }

    def _compute_confidence(self, vec, pred_label) -> float:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec)[0]
            return float(np.max(proba) * 100)

        if hasattr(self.model, "decision_function"):
            # Convert distance-from-hyperplane into a pseudo-probability
            # using a sigmoid so the UI always has a 0-100 number to show.
            score = self.model.decision_function(vec)
            score = score[0] if hasattr(score, "__len__") else score
            sigmoid = 1 / (1 + np.exp(-abs(score)))
            return float(sigmoid * 100)

        return 75.0  # generic fallback if the model exposes neither


# Singleton instance imported by app.py
predictor = NewsPredictor()
