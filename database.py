"""
database.py
------------
Holds the single shared SQLAlchemy instance (`db`) and the Flask-Login
manager (`login_manager`).

They are created here (instead of inside app.py) to avoid circular imports:
models.py needs `db`, and app.py needs both `db` and the models.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

# Where Flask-Login should redirect unauthenticated users
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "warning"
