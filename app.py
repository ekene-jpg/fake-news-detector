"""
Flask application — implements Chapter 3.3.2 System Workflow and
3.3.3(d) Prediction Module.

Routes:
  GET  /          -> input form
  POST /predict   -> runs the pipeline, logs to Prediction_History, shows result
  GET  /history   -> lists past predictions

Run:
  python app.py
Then open http://127.0.0.1:5000 in a browser (works on your phone too, if
your phone is on the same network — see README "Testing on your phone").
"""
import sqlite3
from datetime import datetime

import joblib
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from train_model import clean_text  # reuses the exact preprocessing from training
import fact_check

app = Flask(__name__)
DB_PATH = "news_detection.db"

# --- Load trained artifacts once, at startup ---
try:
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    MODEL_READY = True
except FileNotFoundError:
    model, vectorizer = None, None
    MODEL_READY = False


def predict_news(text):
    """Section 3.3.3(d) Prediction Module."""
    processed = clean_text(text)
    vector = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = round(max(proba) * 100, 1)
    label = "Real News" if prediction == 1 else "Fake News"
    return label, confidence


def log_prediction(input_text, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO Prediction_History (Input_Text, Result, Date) VALUES (?, ?, ?)",
        (input_text, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_ready=MODEL_READY)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return render_template(
            "index.html", model_ready=False,
            error="No trained model found. Run `python train_model.py` first."
        )

    news_text = request.form.get("news_text", "").strip()
    if not news_text:
        return render_template("index.html", model_ready=True, error="Please paste some text to analyze.")

    label, confidence = predict_news(news_text)
    log_prediction(news_text, label)

    # --- External fact-check lookup (Google Fact Check Tools API) ---
    fc_result = fact_check.search_fact_checks(news_text)
    fc_error = fc_result.get("error")
    fc_claims = fc_result.get("claims", [])
    fc_strongest = fact_check.strongest_rating(fc_claims) if not fc_error else None

    final_label, final_note = label, None
    if fc_strongest:
        verdict_type, review = fc_strongest
        if verdict_type == "false":
            final_label = "Fake News"
            final_note = f'Fact-checked FALSE by {review["publisher"]} — this overrides the ML model\'s tone-based read.'
        elif verdict_type == "true":
            final_label = "Real News"
            final_note = f'Fact-checked TRUE by {review["publisher"]} — this overrides the ML model\'s tone-based read.'

    return render_template(
        "index.html",
        model_ready=True,
        submitted_text=news_text,
        result=final_label,
        ml_result=label,
        confidence=confidence,
        fc_error=fc_error,
        fc_claims=fc_claims,
        fc_overridden=final_note
    )


@app.route("/history", methods=["GET"])
def history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT Prediction_ID, Input_Text, Result, Date FROM Prediction_History ORDER BY Prediction_ID DESC LIMIT 50")
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable from your phone on the same Wi-Fi network
    app.run(host="0.0.0.0", port=5000, debug=True)
