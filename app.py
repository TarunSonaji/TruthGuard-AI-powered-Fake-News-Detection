"""
app.py
------
Main Flask application entry point for TruthGuard.

Routes:
    GET  /                Home page (textarea + analyze button)
    GET  /register        Registration form
    POST /register        Create a new user
    GET  /login           Login form
    POST /login           Authenticate a user
    GET  /logout          Log out the current user
    POST /predict         Run a prediction on submitted article text (AJAX)
    GET  /history         Show the logged-in user's prediction history
    GET  /history/export  Export history as CSV
    GET  /about           Project info + model metrics
    POST /api/predict     Public-ish JSON REST API for predictions
"""

import os
import csv
import io
import json
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, Response, session
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)

from config import config_map, Config, BASE_DIR
from database import db, login_manager
from models import User, Prediction
from predict import predictor


def create_app(env: str = "development") -> Flask:
    """Application factory: builds and configures the Flask app instance."""
    app = Flask(__name__)
    app.config.from_object(config_map.get(env, config_map["default"]))

    # Make sure the instance/ and static/uploads/ folders exist
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        db.create_all()

    register_routes(app)
    register_error_handlers(app)
    return app


def register_routes(app: Flask):

    # ---------------------------------------------------------------- HOME
    @app.route("/")
    def index():
        return render_template("index.html", model_ready=predictor.ready)

    # ------------------------------------------------------------ REGISTER
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("index"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")

            # ---- Validation ----
            if not username or not email or not password:
                flash("All fields are required.", "danger")
                return render_template("register.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters long.", "danger")
                return render_template("register.html")

            if password != confirm:
                flash("Passwords do not match.", "danger")
                return render_template("register.html")

            if User.query.filter_by(username=username).first():
                flash("That username is already taken.", "danger")
                return render_template("register.html")

            if User.query.filter_by(email=email).first():
                flash("An account with that email already exists.", "danger")
                return render_template("register.html")

            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    # --------------------------------------------------------------- LOGIN
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("index"))

        if request.method == "POST":
            identifier = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            remember = bool(request.form.get("remember"))

            user = User.query.filter(
                (User.username == identifier) | (User.email == identifier.lower())
            ).first()

            if user and user.check_password(password):
                login_user(user, remember=remember)
                flash(f"Welcome back, {user.username}!", "success")
                next_page = request.args.get("next")
                return redirect(next_page or url_for("index"))

            flash("Invalid username/email or password.", "danger")

        return render_template("login.html")

    # -------------------------------------------------------------- LOGOUT
    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    # ------------------------------------------------------------- PREDICT
    @app.route("/predict", methods=["POST"])
    @login_required
    def predict_route():
        article = request.form.get("article", "").strip()

        if not article:
            return jsonify({"success": False, "error": "Please enter some article text."}), 400

        if len(article) > Config.MAX_ARTICLE_LENGTH:
            return jsonify({
                "success": False,
                "error": f"Article too long (max {Config.MAX_ARTICLE_LENGTH} characters)."
            }), 400

        if len(article.split()) < 4:
            return jsonify({
                "success": False,
                "error": "Please enter a longer piece of text (at least a few words)."
            }), 400

        try:
            result = predictor.predict(article)
        except RuntimeError as e:
            return jsonify({"success": False, "error": str(e)}), 503
        except Exception as e:
            return jsonify({"success": False, "error": f"Prediction failed: {e}"}), 500

        # Save to history
        record = Prediction(
            user_id=current_user.id,
            article=article,
            prediction=result["label"],
            confidence=result["confidence"],
            suspicious_words=", ".join(result["suspicious_words"]),
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            "success": True,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "suspicious_words": result["suspicious_words"],
            "prediction_id": record.id,
            "timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # ------------------------------------------------------------- HISTORY
    @app.route("/history")
    @login_required
    def history():
        search_query = request.args.get("q", "").strip()

        query = Prediction.query.filter_by(user_id=current_user.id)
        if search_query:
            query = query.filter(Prediction.article.ilike(f"%{search_query}%"))

        predictions = query.order_by(Prediction.timestamp.desc()).all()

        total = len(predictions)
        fake_count = sum(1 for p in predictions if p.prediction == "Fake")
        real_count = total - fake_count

        return render_template(
            "history.html",
            predictions=predictions,
            search_query=search_query,
            total=total,
            fake_count=fake_count,
            real_count=real_count,
        )

    # ------------------------------------------------------- HISTORY EXPORT
    @app.route("/history/export")
    @login_required
    def export_history_csv():
        predictions = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.timestamp.desc())
            .all()
        )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Article", "Prediction", "Confidence (%)", "Suspicious Words", "Timestamp"])
        for p in predictions:
            writer.writerow([p.id, p.article, p.prediction, p.confidence, p.suspicious_words, p.timestamp])

        response = Response(output.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=truthguard_history.csv"
        return response

    # ----------------------------------------------------------------ABOUT
    @app.route("/about")
    def about():
        metrics = None
        if os.path.exists(Config.METRICS_PATH):
            with open(Config.METRICS_PATH) as f:
                metrics = json.load(f)
        return render_template("about.html", metrics=metrics)

    # --------------------------------------------------------- REST API
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """
        Public JSON REST API.

        Request:  { "text": "<article text>" }
        Response: { "success": true, "prediction": "Fake", "confidence": 92.3,
                     "suspicious_words": [...] }
        """
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        if not text:
            return jsonify({"success": False, "error": "'text' field is required."}), 400

        try:
            result = predictor.predict(text)
        except RuntimeError as e:
            return jsonify({"success": False, "error": str(e)}), 503
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

        return jsonify({
            "success": True,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "suspicious_words": result["suspicious_words"],
        })

    # ------------------------------------------------------------- PROFILE
    @app.route("/profile")
    @login_required
    def profile():
        total = Prediction.query.filter_by(user_id=current_user.id).count()
        fake = Prediction.query.filter_by(user_id=current_user.id, prediction="Fake").count()
        real = total - fake
        return render_template("profile.html", total=total, fake=fake, real=real)


def register_error_handlers(app: Flask):
    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("500.html"), 500


app = create_app(os.environ.get("FLASK_ENV", "development"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
