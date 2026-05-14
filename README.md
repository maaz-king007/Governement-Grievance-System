# AI-Based Customer Grievance & Response System

**Domain:** Speech and Language Processing (NLP)  
**Team:** Nikhil Kumar (23BAI1264) · Maaz Alam (23BAI1426) · Niharika Ganesh Khopade (23BAI1431)

---

## Project Overview

An end-to-end AI system that automatically handles customer complaints in Hindi/Hinglish using NLP. The system classifies complaints into departments, detects urgency and sentiment, flags abusive language, and generates instant context-aware responses.

## Datasets Used

| Dataset | Source | Usage |
|---------|--------|-------|
| Hinglish Top Dataset | Kaggle | Base Hinglish vocabulary & text patterns |
| Code-Mixed Hateful Hinglish | Kaggle | Abusive/hate speech detection training |
| Indic Sentiment Dataset | Kaggle | Sentiment analysis (positive/negative/neutral) |

> The `data/generate_dataset.py` script generates a synthetic dataset modelled on these datasets for offline use.

## Features

- **Department Classification** — 7 categories (Banking, Telecom, Electricity, Railways, E-Commerce, Healthcare, Government)
- **Sentiment Analysis** — Negative / Neutral / Positive
- **Urgency Detection** — High / Medium / Low (hybrid rule + ML)
- **Hate Speech Detection** — Flags abusive Hinglish language
- **Auto Response Generation** — Context-aware responses per department + urgency
- **Ticket Tracking** — File and track complaints via web UI
- **Admin Dashboard** — Stats, charts, complaint management

## Architecture

```
Input Text (Hindi/Hinglish)
        ↓
  Preprocessor
  ├── Lowercase + clean
  ├── Emoji → sentiment tokens
  ├── Hinglish normalization
  └── Stopword removal
        ↓
  TF-IDF Vectorizer
  (word + char n-grams, sublinear TF)
        ↓
  ┌─────────────────────────────────────┐
  │  LinearSVC  → Department (7 class)  │
  │  LogReg     → Sentiment  (3 class)  │
  │  LinearSVC  → Urgency    (3 class)  │
  │  LogReg     → Abusive    (binary)   │
  └─────────────────────────────────────┘
        ↓
  Response Generator
  (template-based, dept × urgency)
        ↓
  Output + Ticket ID
```

## Quick Start

### Option A — One command
```bash
cd ai_grievance_system
python3 setup_and_run.py
```

### Option B — Step by step
```bash
cd ai_grievance_system

# 1. Generate dataset
python3 data/generate_dataset.py

# 2. Train models
python3 train_models.py

# 3. Start server
python3 app.py
```

Open **http://localhost:5000** in your browser.

## Project Structure

```
ai_grievance_system/
├── app.py                   # Flask web application
├── train_models.py          # Model training script
├── setup_and_run.py         # One-click setup
├── data/
│   └── generate_dataset.py  # Synthetic dataset generator
├── models/                  # Saved models (auto-created)
│   ├── dept_model.pkl
│   ├── sent_model.pkl
│   ├── urg_model.pkl
│   ├── abuse_model.pkl
│   └── metadata.json
├── utils/
│   ├── preprocessor.py      # NLP preprocessing pipeline
│   └── inference.py         # Inference engine
├── templates/
│   ├── index.html
│   ├── submit.html
│   ├── track.html
│   └── dashboard.html
└── static/
    └── css/style.css
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Homepage |
| GET/POST | `/submit` | File a grievance |
| GET | `/track?id=GRV-XXXX` | Track by ticket ID |
| GET | `/dashboard` | Admin dashboard |
| POST | `/api/analyze` | Analyze text (no save) |
| GET | `/api/stats` | Get statistics |
| POST | `/api/update_status` | Update ticket status |

### POST `/api/analyze`
```json
{ "text": "Mera account se paise kat gaye" }
```
Response:
```json
{
  "ticket_id": "GRV-AB12CD34",
  "department": "Banking",
  "dept_confidence": 92,
  "sentiment": "negative",
  "urgency": "medium",
  "is_abusive": false,
  "auto_response": "Aapki banking shikayat register ho gayi hai..."
}
```

## Requirements

- Python 3.8+
- `scikit-learn` — ML models
- `pandas`, `numpy` — Data processing
- `flask` — Web framework

All packages are standard and included in most Python environments.

## Model Performance

Typical accuracy on the generated dataset:
- Department Classification: ~88-95%
- Sentiment Analysis: ~82-90%
- Urgency Detection: ~85-92%
- Abusive Detection: ~90-95%

## Research Gap Addressed

| Gap | Solution |
|-----|----------|
| Limited Hindi NLP support | Custom Hinglish preprocessor + transliteration normalization |
| No urgency detection | Hybrid rule-based + ML urgency classifier |
| Manual department routing | Automated LinearSVC multi-class routing |
| No real-time response | Template-based response generator |
| Hate speech in complaints | Code-mixed abusive language detector |
