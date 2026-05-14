"""
utils/inference.py
Inference engine — loads all trained models and provides predictions
"""

import os
import sys
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessor import preprocess, detect_urgency_keywords, detect_abusive_keywords

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Department-specific auto-response templates
RESPONSE_TEMPLATES = {
    "Banking": {
        "high": "Aapki shikayat hamari priority list mein add kar di gayi hai. Banking team ko 2 ghante mein contact karne ka निर्देश diya gaya hai. Apna account number aur registered mobile number ready rakhein.",
        "medium": "Aapki banking shikayat register ho gayi hai. Hamari team 24 ghante mein aapse contact karegi. Ticket ID save karein reference ke liye.",
        "low": "Aapki shikayat receive ho gayi hai. Banking department 3-5 business days mein resolve karegi. Koi update ke liye is ticket ID se track kar sakte hain.",
    },
    "Telecom": {
        "high": "Network emergency detected! Technical team ko turant alert kar diya gaya hai. Area engineer 4 ghante mein visit karenge. Aapka number escalated queue mein hai.",
        "medium": "Aapka telecom complaint registered hai. Network team 24 ghante mein issue diagnose karegi. Temporary workaround: WiFi calling enable karein settings mein.",
        "low": "Aapki telecom shikayat log ho gayi hai. 48 ghante mein resolution expected hai. Complaint status app mein track kar sakte hain.",
    },
    "Electricity": {
        "high": "Bijli emergency! Electricity board ko immediate alert bhej diya gaya hai. Lineman 2 ghante mein visit karenge. Emergency helpline: 1912",
        "medium": "Electricity complaint register ho gayi hai. Field team 12 ghante mein inspect karegi. Meter reading screenshot save rakhein.",
        "low": "Aapki bijli shikayat receive ho gayi hai. Department 3 working days mein respond karegi. Bill dispute ke liye consumer court ka option bhi available hai.",
    },
    "Railways": {
        "high": "Train emergency complaint registered! Control room ko alert kar diya gaya hai. Station master se turant contact karein ya helpline 139 par call karein.",
        "medium": "Railways complaint log ho gayi hai. 24 ghante mein concerned department respond karegi. PNR number share karein faster resolution ke liye.",
        "low": "Aapki railways shikayat receive ho gayi hai. IRCTC team 5-7 business days mein resolve karegi. Refund cases mein 7-10 days lagते hain.",
    },
    "E-Commerce": {
        "high": "Order issue escalated to priority queue! Seller aur logistics team ko alert kar diya gaya hai. 6 ghante mein resolution update milega.",
        "medium": "Aapki e-commerce complaint register ho gayi hai. Customer care 24 ghante mein contact karegi. Order ID aur photos ready rakhein.",
        "low": "Complaint received. Return/refund policy ke anusaar 5-7 business days mein process hoga. Tracking updates email par milenge.",
    },
    "Healthcare": {
        "high": "Medical emergency complaint! Hospital administration ko turant alert kar diya gaya hai. Patient rights helpline: 1800-180-1104. Aapko 1 ghante mein callback milega.",
        "medium": "Healthcare complaint registered hai. Hospital grievance cell ko forward kar diya gaya hai. 48 ghante mein official response milega.",
        "low": "Aapki healthcare shikayat log ho gayi hai. Medical board 7 days mein review karegi. Documents ready rakhein.",
    },
    "Government": {
        "high": "Government service complaint escalated! Concerned officer ko immediate notice bhej diya gaya hai. Bribe complaint ke liye: Anti-Corruption Helpline 1064",
        "medium": "Sarkari shikayat darj ho gayi hai. Concerned department 72 ghante mein respond karne ke liye bound hai (RTI provisions ke anusaar).",
        "low": "Aapki government service complaint receive ho gayi hai. 30 days mein resolution mandatory hai government norms ke anusaar.",
    },
}

ABUSIVE_WARNING = (
    "⚠️ Aapki complaint mein inappropriate language detect hui hai. "
    "Hamari team aapki madad karne ke liye tayyar hai, lekin civil language "
    "mein communicate karna zaroori hai. Aapki core complaint process ki ja rahi hai."
)


class GrievancePredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load_models(self):
        if self._loaded:
            return
        print("Loading models...")
        with open(f"{MODELS_DIR}/dept_model.pkl", "rb") as f:
            self.dept_model = pickle.load(f)
        with open(f"{MODELS_DIR}/dept_le.pkl", "rb") as f:
            self.dept_le = pickle.load(f)
        with open(f"{MODELS_DIR}/sent_model.pkl", "rb") as f:
            self.sent_model = pickle.load(f)
        with open(f"{MODELS_DIR}/sent_le.pkl", "rb") as f:
            self.sent_le = pickle.load(f)
        with open(f"{MODELS_DIR}/urg_model.pkl", "rb") as f:
            self.urg_model = pickle.load(f)
        with open(f"{MODELS_DIR}/urg_le.pkl", "rb") as f:
            self.urg_le = pickle.load(f)
        with open(f"{MODELS_DIR}/abuse_model.pkl", "rb") as f:
            self.abuse_model = pickle.load(f)
        with open(f"{MODELS_DIR}/metadata.json", "r") as f:
            self.meta = json.load(f)
        self._loaded = True
        print("Models loaded ✓")

    def predict(self, raw_text: str) -> dict:
        self.load_models()

        clean = preprocess(raw_text)

        # ── Department ────────────────────────────────────────────────────────
        dept_enc = self.dept_model.predict([clean])[0]
        department = self.dept_le.inverse_transform([dept_enc])[0]

        # Confidence via decision function (LinearSVC)
        try:
            dept_scores = self.dept_model.decision_function([clean])[0]
            dept_conf = float(max(dept_scores))
            dept_conf_pct = min(99, max(50, int(50 + dept_conf * 15)))
        except Exception:
            dept_conf_pct = 85

        # ── Sentiment ─────────────────────────────────────────────────────────
        sent_enc = self.sent_model.predict([clean])[0]
        sentiment = self.sent_le.inverse_transform([sent_enc])[0]

        try:
            sent_proba = self.sent_model.predict_proba([clean])[0]
            sent_conf = int(max(sent_proba) * 100)
        except Exception:
            sent_conf = 80

        # ── Urgency ───────────────────────────────────────────────────────────
        urg_enc = self.urg_model.predict([clean])[0]
        urgency_ml = self.urg_le.inverse_transform([urg_enc])[0]

        # Hybrid: rule-based overrides if keywords detected
        urgency_rule = detect_urgency_keywords(raw_text)
        urgency = urgency_rule if urgency_rule == "high" else urgency_ml

        # ── Abusive ───────────────────────────────────────────────────────────
        is_abusive_ml = bool(self.abuse_model.predict([clean])[0])
        is_abusive_rule = detect_abusive_keywords(raw_text)
        is_abusive = is_abusive_ml or is_abusive_rule

        # ── Response generation ───────────────────────────────────────────────
        templates = RESPONSE_TEMPLATES.get(department, RESPONSE_TEMPLATES["Government"])
        auto_response = templates.get(urgency, templates["low"])

        if is_abusive:
            auto_response = ABUSIVE_WARNING + "\n\n" + auto_response

        # ── Build result ──────────────────────────────────────────────────────
        import random, string
        ticket_id = "GRV-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

        return {
            "ticket_id": ticket_id,
            "department": department,
            "dept_confidence": dept_conf_pct,
            "sentiment": sentiment,
            "sent_confidence": sent_conf,
            "urgency": urgency,
            "is_abusive": is_abusive,
            "auto_response": auto_response,
            "preprocessed_text": clean,
        }


# Singleton
predictor = GrievancePredictor()
