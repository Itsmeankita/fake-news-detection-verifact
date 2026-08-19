"""
VeriFact Pro — Fake News Detection System (extended edition)
"""
import csv
import io
import json
import logging
import os
import secrets
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

import joblib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                    request, send_file, url_for)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                          login_user, logout_user)
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from preprocess import clean_text  # noqa: E402
from explain import top_contributing_words, global_top_words  # noqa: E402
from factcheck import get_related_factchecks  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "verifact.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ------------------------------------------------------------------ logging --
_handler = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=500_000, backupCount=2)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
app.logger.addHandler(_handler)
app.logger.setLevel(logging.INFO)

# ---------------------------------------------------------------- rate limits --
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per hour"],
                   storage_uri="memory://")

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to view that page."


# ---------------------------------------------------------------- models --
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    api_key = db.Column(db.String(64), unique=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class HistoryEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    text_snippet = db.Column(db.String(300))
    prediction = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    source = db.Column(db.String(20), default="text")  # text / url / batch
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()


# ---------------------------------------------------------- model loading --
model = None
vectorizer = None
metrics = {}


def load_artifacts():
    global model, vectorizer, metrics
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)


load_artifacts()


def run_prediction(text):
    """Shared prediction logic used by every detection route."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        confidence = float(max(proba)) * 100
        fake_prob = float(proba[classes.index("FAKE")]) * 100
        real_prob = float(proba[classes.index("REAL")]) * 100
    else:
        score = model.decision_function(vec)[0]
        confidence = min(99.0, 50 + abs(score) * 12)
        fake_prob = confidence if prediction == "FAKE" else 100 - confidence
        real_prob = 100 - fake_prob

    fake_words, real_words = top_contributing_words(model, vectorizer, vec)
    factchecks = get_related_factchecks(text)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "real_probability": round(real_prob, 1),
        "fake_probability": round(fake_prob, 1),
        "word_count": len(text.split()),
        "char_count": len(text),
        "model_used": metrics.get("deployed_model", "Logistic Regression"),
        "fake_indicator_words": fake_words,
        "real_indicator_words": real_words,
        "factchecks": factchecks,
    }


def save_history(text, result, source="text"):
    if current_user.is_authenticated:
        entry = HistoryEntry(
            user_id=current_user.id,
            text_snippet=text[:280],
            prediction=result["prediction"],
            confidence=result["confidence"],
            source=source,
        )
        db.session.add(entry)
        db.session.commit()


# ------------------------------------------------------------- page routes --
@app.route("/")
def home():
    return render_template("index.html", active="home", metrics=metrics)


@app.route("/detect")
def detect():
    model_ready = model is not None and vectorizer is not None
    return render_template("detect.html", active="detect", model_ready=model_ready,
                            metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html", active="about", metrics=metrics)


@app.route("/analytics")
def analytics():
    model_ready = model is not None and vectorizer is not None
    has_wordclouds = os.path.exists(os.path.join(BASE_DIR, "static", "img", "wordcloud_real.png"))
    fake_words, real_words = ([], [])
    if model_ready:
        fake_words, real_words = global_top_words(model, vectorizer)
    return render_template("analytics.html", active="analytics",
                            metrics=metrics, model_ready=model_ready,
                            has_wordclouds=has_wordclouds,
                            fake_words=fake_words, real_words=real_words)


@app.route("/compare")
def compare_page():
    model_ready = model is not None and vectorizer is not None
    return render_template("compare.html", active="compare", model_ready=model_ready)


@app.route("/quiz")
def quiz():
    return render_template("quiz.html", active="quiz")


@app.route("/faq")
def faq():
    return render_template("faq.html", active="faq")


# ------------------------------------------------------------- auth routes --
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or len(password) < 6:
            flash("Please fill every field. Password must be at least 6 characters.", "error")
            return render_template("signup.html", active="signup")

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("That username or email is already registered.", "error")
            return render_template("signup.html", active="signup")

        user = User(username=username, email=email, api_key=secrets.token_hex(20))
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Account created — welcome to VeriFact Pro!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html", active="signup")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.", "error")
    return render_template("login.html", active="login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    entries = (HistoryEntry.query.filter_by(user_id=current_user.id)
               .order_by(HistoryEntry.created_at.desc()).limit(50).all())
    real_count = sum(1 for e in entries if e.prediction == "REAL")
    fake_count = sum(1 for e in entries if e.prediction == "FAKE")
    return render_template("dashboard.html", active="dashboard", entries=entries,
                            real_count=real_count, fake_count=fake_count)


@app.route("/admin")
@login_required
def admin():
    if not current_user.is_admin:
        flash("Admin access only.", "error")
        return redirect(url_for("home"))
    users = User.query.order_by(User.created_at.desc()).all()
    total_checks = HistoryEntry.query.count()
    return render_template("admin.html", active="admin", users=users,
                            total_checks=total_checks, metrics=metrics)


# --------------------------------------------------------------- API routes --
@app.after_request
def add_cors_headers(response):
    """Allows the companion browser extension (a separate origin) to call
    the prediction API. Scoped to /api/ only — page routes are untouched."""
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/predict", methods=["POST"])
@limiter.limit("30 per minute")
def api_predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet. Run "
                                  "'python model/train_model.py' first."}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 20:
        return jsonify({"error": "Please paste at least a full sentence "
                                  "(20+ characters) of news text to analyze."}), 400

    result = run_prediction(text)
    save_history(text, result, source="text")
    app.logger.info(f"predict text_len={len(text)} verdict={result['prediction']} conf={result['confidence']}")
    return jsonify(result)


@app.route("/api/predict-url", methods=["POST"])
def api_predict_url():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet."}), 503

    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url.startswith("http"):
        return jsonify({"error": "Please enter a valid URL starting with http(s)://"}), 400

    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        article_text = " ".join(paragraphs)
    except Exception:
        return jsonify({"error": "Couldn't fetch that URL. The site may block "
                                  "automated requests, or the link may be invalid."}), 400

    if len(article_text) < 50:
        return jsonify({"error": "Couldn't extract enough article text from that "
                                  "page. Try pasting the text directly instead."}), 400

    result = run_prediction(article_text)
    result["extracted_preview"] = article_text[:400]
    save_history(article_text, result, source="url")
    return jsonify(result)


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        stream = io.StringIO(file.stream.read().decode("utf-8"))
        reader = csv.DictReader(stream)
        if "text" not in (reader.fieldnames or []):
            return jsonify({"error": "CSV must have a column named 'text'."}), 400

        results = []
        for row in reader:
            text = (row.get("text") or "").strip()
            if len(text) < 20:
                continue
            r = run_prediction(text)
            results.append({
                "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
                "prediction": r["prediction"],
                "confidence": r["confidence"],
            })
            save_history(text, r, source="batch")
            if len(results) >= 200:  # sane cap for a demo
                break
    except Exception as e:
        return jsonify({"error": f"Couldn't process that file: {e}"}), 400

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet."}), 503

    data = request.get_json(silent=True) or {}
    text_a = (data.get("text_a") or "").strip()
    text_b = (data.get("text_b") or "").strip()
    if len(text_a) < 20 or len(text_b) < 20:
        return jsonify({"error": "Both articles need at least 20 characters."}), 400

    return jsonify({"result_a": run_prediction(text_a), "result_b": run_prediction(text_b)})


@app.route("/api/export-pdf", methods=["POST"])
def api_export_pdf():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = data.get("result", {})

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("VeriFact — Detection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 18),
        Paragraph(f"<b>Verdict:</b> {result.get('prediction', '—')}", styles["Heading2"]),
        Paragraph(f"<b>Confidence:</b> {result.get('confidence', '—')}%", styles["Normal"]),
        Paragraph(f"<b>Model used:</b> {result.get('model_used', '—')}", styles["Normal"]),
        Spacer(1, 18),
        Paragraph("<b>Analyzed text:</b>", styles["Heading3"]),
        Paragraph(text[:3000].replace("\n", "<br/>"), styles["Normal"]),
    ]
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                      download_name="verifact-report.pdf")


# ------------------------------------------------------------ new: system --
@app.route("/api/health")
def api_health():
    """Simple health check — useful for uptime monitors and deployment
    platforms (Render/Railway ping this to know the app is alive)."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.route("/api/model-info")
def api_model_info():
    """Machine-readable summary of the deployed model — useful for anyone
    integrating with the API, and doubles as lightweight API documentation."""
    return jsonify({
        "deployed_model": metrics.get("deployed_model"),
        "accuracy": metrics.get("deployed_metrics", {}).get("accuracy"),
        "trained_on": metrics.get("dataset_source"),
        "training_articles": metrics.get("total_articles"),
        "vocabulary_size": metrics.get("vocabulary_size"),
        "endpoints": {
            "POST /api/predict": "body: {text}. Returns verdict + confidence + explainability.",
            "POST /api/predict-url": "body: {url}. Scrapes and analyzes an article.",
            "POST /api/predict-batch": "multipart file upload, CSV with a 'text' column.",
            "POST /api/compare": "body: {text_a, text_b}. Compares two articles.",
            "POST /api/export-pdf": "body: {text, result}. Returns a PDF report.",
            "GET /api/health": "Liveness check.",
            "GET /api/model-info": "This endpoint.",
        },
    })


# ------------------------------------------------------- new: account mgmt --
@app.route("/dashboard/change-password", methods=["POST"])
@login_required
def change_password():
    current_pw = request.form.get("current_password", "")
    new_pw = request.form.get("new_password", "")

    if not current_user.check_password(current_pw):
        flash("Current password is incorrect.", "error")
    elif len(new_pw) < 6:
        flash("New password must be at least 6 characters.", "error")
    else:
        current_user.set_password(new_pw)
        db.session.commit()
        flash("Password updated.", "success")
    return redirect(url_for("dashboard"))


@app.route("/dashboard/export-history")
@login_required
def export_history():
    """Downloads the logged-in user's full detection history as a CSV."""
    entries = (HistoryEntry.query.filter_by(user_id=current_user.id)
               .order_by(HistoryEntry.created_at.desc()).all())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "text_snippet", "prediction", "confidence", "source"])
    for e in entries:
        writer.writerow([e.created_at.isoformat(), e.text_snippet, e.prediction,
                          e.confidence, e.source])

    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=verifact-history.csv"}
    )


@app.route("/dashboard/delete-entry/<int:entry_id>", methods=["POST"])
@login_required
def delete_history_entry(entry_id):
    entry = db.session.get(HistoryEntry, entry_id)
    if entry and entry.user_id == current_user.id:
        db.session.delete(entry)
        db.session.commit()
        flash("Entry deleted.", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
