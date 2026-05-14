"""
app.py — Flask Web Application
AI-Based Customer Grievance and Response System
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config["SECRET_KEY"] = "grievance_system_2024"

DB_PATH = "grievances.db"


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS grievances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            name TEXT,
            email TEXT,
            phone TEXT,
            raw_text TEXT,
            department TEXT,
            dept_confidence INTEGER,
            sentiment TEXT,
            urgency TEXT,
            is_abusive INTEGER,
            auto_response TEXT,
            status TEXT DEFAULT 'Open',
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_grievance(data: dict, name: str, email: str, phone: str, raw_text: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO grievances
        (ticket_id, name, email, phone, raw_text, department, dept_confidence,
         sentiment, urgency, is_abusive, auto_response, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open', ?)
    """, (
        data["ticket_id"], name, email, phone, raw_text,
        data["department"], data["dept_confidence"],
        data["sentiment"], data["urgency"],
        int(data["is_abusive"]), data["auto_response"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def get_all_grievances():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM grievances ORDER BY created_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_grievance_by_ticket(ticket_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM grievances WHERE ticket_id = ?", (ticket_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM grievances").fetchone()[0]
    by_dept = dict(c.execute(
        "SELECT department, COUNT(*) FROM grievances GROUP BY department"
    ).fetchall())
    by_urgency = dict(c.execute(
        "SELECT urgency, COUNT(*) FROM grievances GROUP BY urgency"
    ).fetchall())
    by_sentiment = dict(c.execute(
        "SELECT sentiment, COUNT(*) FROM grievances GROUP BY sentiment"
    ).fetchall())
    by_status = dict(c.execute(
        "SELECT status, COUNT(*) FROM grievances GROUP BY status"
    ).fetchall())
    conn.close()
    return {
        "total": total,
        "by_dept": by_dept,
        "by_urgency": by_urgency,
        "by_sentiment": by_sentiment,
        "by_status": by_status,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "GET":
        return render_template("submit.html")

    data = request.get_json()
    name = data.get("name", "Anonymous")
    email = data.get("email", "")
    phone = data.get("phone", "")
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Grievance text required"}), 400

    from utils.inference import predictor
    result = predictor.predict(text)
    save_grievance(result, name, email, phone, text)

    return jsonify(result)


@app.route("/track")
def track():
    ticket_id = request.args.get("id", "")
    grievance = None
    if ticket_id:
        grievance = get_grievance_by_ticket(ticket_id)
    return render_template("track.html", grievance=grievance, ticket_id=ticket_id)


@app.route("/dashboard")
def dashboard():
    grievances = get_all_grievances()
    stats = get_stats()
    meta_path = "models/metadata.json"
    model_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_meta = json.load(f)
    return render_template("dashboard.html",
                           grievances=grievances,
                           stats=stats,
                           model_meta=model_meta)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """REST API endpoint for analysis without saving."""
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400
    from utils.inference import predictor
    result = predictor.predict(text)
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/update_status", methods=["POST"])
def update_status():
    data = request.get_json()
    ticket_id = data.get("ticket_id")
    status = data.get("status")
    if ticket_id and status:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE grievances SET status=? WHERE ticket_id=?",
                     (status, ticket_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    return jsonify({"error": "Invalid data"}), 400


if __name__ == "__main__":
    init_db()
    print("🚀 AI Grievance System starting...")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
