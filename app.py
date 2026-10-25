
import os
import sqlite3
from functools import wraps
from contextlib import contextmanager
from pathlib import Path

import joblib
from flask import (
    Flask, flash, redirect, render_template,
    request, session, url_for
)
from werkzeug.security import (
    check_password_hash, generate_password_hash
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "users.db"
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

if not app.config["SECRET_KEY"]:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set."
    )

app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


@contextmanager
def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access the detector.", "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def classify_news(text):
    """Classify news using the saved TF-IDF + ML pipeline."""

    if not MODEL_PATH.exists():
        return None, None, "missing model"

    try:
        model = joblib.load(MODEL_PATH)
        prediction = int(model.predict([text])[0])
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            confidence = round(float(max(probabilities)) * 100, 1)

        if prediction == 1:
            result = "Likely Real"
        elif prediction == 0:
            result = "Likely Fake"
        else:
            return None, None, "model error"

        return result, confidence, "saved model"

    except Exception:
        app.logger.exception("Could not load or use model.joblib")
        return None, None, "model error"


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill in all required fields.", "error")
            return render_template("register.html")

        if len(name) > 80:
            flash("Name must be 80 characters or fewer.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        try:
            with get_db() as db:
                cursor = db.execute(
                    """INSERT INTO users (name, email, password_hash)
                       VALUES (?, ?, ?)""",
                    (name, email, generate_password_hash(password))
                )
                user_id = cursor.lastrowid

        except sqlite3.IntegrityError:
            flash(
                "An account with this email already exists. Please log in.",
                "error"
            )
            return render_template("register.html")

        session.clear()
        session["user_id"] = user_id
        session["user_name"] = name

        flash("Account created successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db() as db:
            user = db.execute(
                """SELECT id, name, email, password_hash
                   FROM users WHERE email = ?""",
                (email,)
            ).fetchone()

        if user is None or not check_password_hash(
            user["password_hash"], password
        ):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]

        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    model_available = MODEL_PATH.exists()

    return render_template(
        "dashboard.html",
        model_available=model_available
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    news_text = request.form.get("news_text", "").strip()

    if len(news_text) < 20:
        flash(
            "Please enter a news article or headline with at least 20 characters.",
            "error"
        )
        return redirect(url_for("dashboard"))

    if len(news_text) > 20000:
        flash("Please keep the text under 20,000 characters.", "error")
        return redirect(url_for("dashboard"))

    result, confidence, status = classify_news(news_text)

    if status == "missing model":
        flash("Model file is missing. Run train_model.py first.", "error")
        return render_template(
            "dashboard.html",
            model_available=False,
            news_text=news_text
        )

    if status == "model error":
        flash(
            "The model could not be loaded. Check the terminal for details.",
            "error"
        )
        return render_template(
            "dashboard.html",
            model_available=False,
            news_text=news_text
        )

    return render_template(
        "dashboard.html",
        model_available=True,
        news_text=news_text,
        result=result,
        confidence=confidence
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.errorhandler(413)
def too_large(_error):
    flash("The submitted request is too large.", "error")
    return redirect(url_for("dashboard"))


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


init_db()

if __name__ == "__main__":
    app.run(debug=True)
