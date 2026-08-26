"""
models.py
---------
Database models for TruthGuard.

Tables
------
User        : registered application users (Flask-Login compatible)
Prediction  : history of every article a user has submitted for analysis
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from database import db, login_manager


class User(UserMixin, db.Model):
    """Represents a registered user of the application."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    # One user can have many predictions
    predictions = db.relationship(
        "Prediction", backref="author", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, raw_password: str) -> None:
        """Hash and store the user's password. Never store plain text."""
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Verify a plain-text password against the stored hash."""
        return check_password_hash(self.password_hash, raw_password)

    def __repr__(self):
        return f"<User {self.username}>"


class Prediction(db.Model):
    """Represents a single fake/real news prediction made by a user."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    article = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)  # "Fake" or "Real"
    confidence = db.Column(db.Float, nullable=False)        # 0 - 100
    suspicious_words = db.Column(db.Text, nullable=True)     # comma separated
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        """Serialize for JSON responses (used by /api and history search)."""
        return {
            "id": self.id,
            "article": self.article,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "suspicious_words": self.suspicious_words,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def __repr__(self):
        return f"<Prediction {self.id} - {self.prediction}>"


@login_manager.user_loader
def load_user(user_id):
    """Required by Flask-Login: reload a user object from the session id."""
    return User.query.get(int(user_id))
