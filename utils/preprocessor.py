"""
utils/preprocessor.py
Hinglish / Code-Mixed text preprocessing pipeline
"""

import re
import string


# Common Hinglish stopwords (Roman Urdu/Hindi)
HINGLISH_STOPWORDS = {
    "hai", "hain", "tha", "thi", "the", "ho", "hoga", "hogi",
    "ka", "ki", "ke", "ko", "se", "par", "pe", "mein", "me",
    "aur", "ya", "yaar", "bhi", "sirf", "toh", "to", "koi",
    "kuch", "mere", "mera", "meri", "main", "hum", "aap", "wo",
    "woh", "yeh", "ye", "is", "us", "ek", "iss", "ne", "na",
    "nahi", "nhi", "hi", "ab", "kab", "kya", "kyun", "kyunki",
    "lekin", "magar", "phir", "fir", "warna", "tabhi", "jab",
    "jo", "jis", "jin", "jab", "jabse", "agar", "agar", "bahut",
    "itna", "utna", "kitna", "bilkul", "baar", "din",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet",
    "for", "in", "on", "at", "to", "by", "of", "with",
    "not", "no", "very", "just", "also",
}

# Transliteration normalizations (common misspellings / variants)
NORMALIZATIONS = {
    "nhi": "nahi",
    "ni": "nahi",
    "plz": "please",
    "pls": "please",
    "hlp": "help",
    "thk": "theek",
    "thik": "theek",
    "bhut": "bahut",
    "bht": "bahut",
    "acha": "accha",
    "karo": "karo",
    "krna": "karna",
    "krta": "karta",
    "gya": "gaya",
    "hua": "hua",
    "hn": "hain",
    "h": "hai",
    "kr": "kar",
    "rha": "raha",
    "rhi": "rahi",
    "pta": "pata",
    "pta nhi": "pata nahi",
    "bt": "baat",
    "bta": "bata",
    "smjh": "samajh",
    "lga": "laga",
    "mil": "mila",
}

EMOJI_SENTIMENT = {
    "😠": " negative angry ",
    "😡": " negative angry ",
    "😤": " negative frustrated ",
    "😢": " negative sad ",
    "😭": " negative very_sad ",
    "🤬": " negative abusive ",
    "👎": " negative disapprove ",
    "😊": " positive happy ",
    "🙏": " request polite ",
    "😞": " negative disappointed ",
    "💔": " negative hurt ",
    "✅": " positive resolved ",
}


def replace_emojis(text: str) -> str:
    for emoji, sentiment in EMOJI_SENTIMENT.items():
        text = text.replace(emoji, sentiment)
    return text


def normalize_hinglish(text: str) -> str:
    words = text.split()
    normalized = []
    for w in words:
        normalized.append(NORMALIZATIONS.get(w.lower(), w))
    return " ".join(normalized)


def clean_text(text: str) -> str:
    """Full preprocessing pipeline."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Replace emojis with sentiment tokens
    text = replace_emojis(text)

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " url ", text)

    # 4. Remove special characters but keep alphanumeric + spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Normalize common Hinglish spellings
    text = normalize_hinglish(text)

    return text


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [t for t in tokens if t not in HINGLISH_STOPWORDS and len(t) > 1]
    return " ".join(filtered)


def preprocess(text: str, remove_stops: bool = True) -> str:
    text = clean_text(text)
    if remove_stops:
        text = remove_stopwords(text)
    return text


def detect_urgency_keywords(text: str) -> str:
    """Rule-based urgency boost on raw text."""
    text_lower = text.lower()
    high_kw = [
        "emergency", "urgent", "turant", "jaldi", "abhi", "hospital",
        "ambulance", "accident", "death", "mar", "serious", "critical",
        "life", "jaan", "khatra", "12 ghante", "24 ghante",
    ]
    medium_kw = [
        "problem", "issue", "pareshaan", "frustrated", "disappointed",
        "waiting", "ruk", "band", "fail", "error",
    ]
    for kw in high_kw:
        if kw in text_lower:
            return "high"
    for kw in medium_kw:
        if kw in text_lower:
            return "medium"
    return "low"


def detect_abusive_keywords(text: str) -> bool:
    """Rule-based abusive/hateful language detection."""
    text_lower = text.lower()
    abusive_kw = [
        "chor", "bakwaas", "bewakoof", "ullu", "gadha", "harami",
        "kamina", "kamine", "fraud", "dhoka", "loot", "cheat",
        "idiot", "stupid", "useless", "worthless", "scam", "scammer",
        "barbaad", "destroy", "sue", "court", "police",
    ]
    for kw in abusive_kw:
        if kw in text_lower:
            return True
    return False
