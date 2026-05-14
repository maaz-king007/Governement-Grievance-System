"""
Synthetic dataset generator mimicking:
- Hinglish Top Dataset (Kaggle)
- Code-Mixed Hateful Hinglish dataset
- Indic Sentiment dataset
"""

import pandas as pd
import random
import os

random.seed(42)

TEMPLATES = {
    "Banking": [
        "mera account se paise kat gaye bina kisi reason ke, please help karo",
        "ATM se cash nahi nikla but account se deduct ho gaya yaar",
        "Net banking band ho gaya hai, login nahi ho raha bilkul bhi",
        "Mera cheque bounce ho gaya unnecessarily, koi reason nahi diya",
        "Credit card ka bill galat hai, extra charges laga diye unhone",
        "Loan EMI double deduct ho gayi is month, very frustrating",
        "Bank branch mein koi sunata hi nahi, staff bahut rude hai",
        "FD maturity ke paise abhi tak nahi mile, kitna wait karein",
        "UPI transaction fail ho gayi par paise cut ho gaye account se",
        "Mobile banking app crash ho rahi hai baar baar, useless app",
        "Mera savings account freeze kar diya without any notice",
        "Interest rate galat calculate ki gayi meri FD par",
    ],
    "Telecom": [
        "Network nahi aa raha ghar pe, signal bilkul zero hai",
        "Recharge kiya par balance nahi aaya, paisa waste ho gaya",
        "Internet speed bahut slow hai, 2G jaisi feel ho rahi hai",
        "Call drop ho rahi hai baar baar, baat nahi ho pa rahi",
        "Data pack khatam nahi hua tha fir bhi band kar diya",
        "Customer care number pe koi nahi uthata, 1 ghante se hold par hoon",
        "Outgoing calls band kar di bina kisi warning ke",
        "Roaming charges bahut zyada laga diye trip mein",
        "SMS nahi aa raha important messages, very problematic",
        "5G plan liya par 4G bhi nahi milta area mein",
        "OTP nahi aa raha important verification ke liye",
        "Port request reject kar diya bina koi reason bataye",
    ],
    "Electricity": [
        "Bijli bill bahut zyada aaya is baar, double ho gaya",
        "Power cut ho rahi hai roz roz, kab theek hoga",
        "Meter reading galat hai, humne check kiya",
        "New connection ke liye 3 mahine se wait kar rahe hain",
        "Transformer kharab hai colony mein, koi nahi aa raha fix karne",
        "Voltage fluctuation se ghar ka saman kharab ho gaya",
        "Bill online pay kiya par receipt nahi mili abhi tak",
        "Supply 12 ghante se band hai, koi update nahi de raha",
        "Lineman bribe maang raha hai connection ke liye, wrong hai ye",
        "Smart meter galat reading de raha hai",
    ],
    "Railways": [
        "Train 5 ghante late hai, koi announcement nahi ki",
        "Ticket confirm nahi hua par paise kat gaye account se",
        "AC coach mein AC band hai, bahut garmi ho rahi hai",
        "Khana bahut ganda tha train mein, quality zero hai",
        "Reservation coach mein general passengers aa gaye",
        "Station par porter ne bahut zyada charge kiya",
        "Refund nahi mila cancelled ticket ka, 2 mahine ho gaye",
        "Platform ticket machine kaam nahi kar rahi",
        "Waiting room ki haalat bahut kharab hai, cleaning nahi hoti",
        "Train mein theft ho gaya, TTE kuch nahi kar raha",
    ],
    "E-Commerce": [
        "Order deliver nahi hua par delivered show ho raha hai app mein",
        "Product damaged mila box kholne par, quality check nahi tha",
        "Return request reject ho gaya bina reason ke",
        "Refund pending hai 15 din se, paisa wapas karo",
        "Wrong product bheja gaya, size aur color dono galat",
        "Fake product mila, original nahi tha bilkul",
        "Delivery boy ne rude behave kiya, acceptable nahi",
        "Discount nahi mila jo website par show ho raha tha",
        "Order cancel kar diya aapne meri permission bina",
        "COD order mein bhi advance payment maang rahe ho, why",
        "Product quality bahut kharab hai, waste of money",
        "Seller ne galat description diya product ka",
    ],
    "Healthcare": [
        "Doctor ne appointment cancel kar diya last minute par",
        "Hospital bill mein galat charges add kiye gaye hain",
        "Medicine out of stock hai, alternative nahi bataya",
        "Ambulance bahut late aayi emergency mein, dangerous tha",
        "Lab report galat aayi, second test mein alag results",
        "Insurance claim reject ho gaya without valid reason",
        "Doctor ne theek se examine nahi kiya, jaldi mein tha",
        "Ward mein cleanliness bahut kharab hai, infection risk",
        "Online appointment system kaam nahi kar raha",
        "Medicines expire ho chuki thi jo di gayi",
    ],
    "Government": [
        "Aadhaar update nahi ho raha, portal crash ho raha hai",
        "Passport application status update nahi ho raha weeks se",
        "Income certificate ke liye 2 mahine se chakkar laga raha hoon",
        "Sarkari officer bribe maang raha hai kaam karne ke liye",
        "RTI application ka jawab nahi mila 30 din mein",
        "Pension ruk gayi hai, senior citizen ko bahut problem hai",
        "Property registration mein delays ho rahi hain",
        "Caste certificate reject ho gaya without reason",
        "PAN card address update nahi ho rahi online",
        "Scholarship nahi mili jo milni chahiye thi students ko",
    ],
}

SENTIMENT_NEGATIVE_PHRASES = [
    "bahut bura laga", "bilkul galat hai", "shame on you",
    "worst service", "pareshaan kar diya", "disgusting",
    "kab theek hoga", "unacceptable", "very disappointed",
    "bahut gussa aa raha hai", "ye toh zabardasti hai",
    "ek baar bhi sahi se nahi kiya", "sab jhooth bolte ho",
    "itna bura experience pehle kabhi nahi tha", "bahut nirasha hui",
]

SENTIMENT_NEUTRAL_PHRASES = [
    "please help karo", "solution chahiye",
    "ummeed hai theek hoga", "request hai", "please resolve",
    "inform kar dena jab theek ho", "let me know the status",
    "update chahiye complaint ka", "mujhe jaankari chahiye",
    "koi bata sakta hai kya", "jab possible ho theek kar dena",
    "dekhte hain kya hota hai", "complaint darz kar raha hoon",
]

SENTIMENT_POSITIVE_PHRASES = [
    "aasha hai jaldi theek hoga", "bahut shukriya pehle se help ke liye",
    "team pe bharosa hai", "confident hoon aap resolve kar lenge",
    "pichle baar bahut achha support mila tha",
    "ummeed hai is baar bhi theek kar denge",
    "aap log bahut helpful hote hain usually",
]

ABUSIVE_PHRASES = [
    "bakwaas company hai", "chor hain sab", "bewakoof log",
    "barbaad kar diya", "loot rahe ho", "dhoka hai ye",
    "ullu banate ho customers ko", "fraud system hai ye",
    "cheat kar rahe ho khullam khulla", "yeh toh dakaiti hai",
]

HIGH_URGENCY_PHRASES = [
    "emergency hai abhi turant help chahiye",
    "jaan ka khatra hai please urgent basis par",
    "bahut serious matter hai ye turant resolve karo",
    "ambulance late aayi aur patient critical tha",
    "bijli 12 ghante se gai hai emergency hai",
    "bimar bachha hai ghar mein urgent help",
    "accident ho gaya hospital se baat karo",
    "critical situation hai please help karo abhi",
]

MEDIUM_URGENCY_PHRASES = [
    "kaafi din se yeh problem chal rahi hai",
    "2-3 din mein resolve karo please",
    "bohot pareshaan hoon is issue se please help",
    "jaldi se theek karo yaar office mein problem ho rahi hai",
    "important kaam ruk gaya is wajah se please",
    "har din aata hoon office mein issue hai",
]

LOW_URGENCY_PHRASES = [
    "jab time mile tab resolve kar dena",
    "koi urgent nahi hai but issue hai please dekh lena",
    "agar possible ho to theek kar dena",
    "slowly but theek kar dena please",
    "no rush but complaint darz kar raha hoon",
]

URGENCY_HIGH_MARKERS = [
    "emergency", "jaan ka khatra", "abhi turant", "urgent",
    "hospital", "ambulance", "12 ghante", "serious",
    "critical", "accident", "bimar", "bahut serious",
]


def make_row(text, dept, sentiment, urgency, is_abusive=0):
    return {
        "text": text.strip(),
        "department": dept,
        "sentiment": sentiment,
        "urgency": urgency,
        "is_abusive": is_abusive,
    }


def generate():
    rows = []

    for dept, templates in TEMPLATES.items():
        for t in templates:
            # --- NEGATIVE sentiment ---
            for phrase in random.sample(SENTIMENT_NEGATIVE_PHRASES, 3):
                urg = "medium" if random.random() < 0.5 else "low"
                rows.append(make_row(t + " " + phrase, dept, "negative", urg))

            # --- NEUTRAL sentiment ---
            for phrase in random.sample(SENTIMENT_NEUTRAL_PHRASES, 3):
                urg = "low" if random.random() < 0.6 else "medium"
                rows.append(make_row(t + " " + phrase, dept, "neutral", urg))

            # --- POSITIVE sentiment ---
            for phrase in random.sample(SENTIMENT_POSITIVE_PHRASES, 2):
                rows.append(make_row(t + " " + phrase, dept, "positive", "low"))

            # --- ABUSIVE (always negative) ---
            if random.random() < 0.4:
                ab = random.choice(ABUSIVE_PHRASES)
                rows.append(make_row(t + " " + ab, dept, "negative", "medium", 1))

        # --- Explicit HIGH urgency samples per dept ---
        for phrase in HIGH_URGENCY_PHRASES:
            base = random.choice(templates)
            rows.append(make_row(base + " " + phrase, dept, "negative", "high"))

        # --- Explicit MEDIUM urgency ---
        for phrase in MEDIUM_URGENCY_PHRASES:
            base = random.choice(templates)
            sent = random.choice(["negative", "neutral"])
            rows.append(make_row(base + " " + phrase, dept, sent, "medium"))

        # --- Explicit LOW urgency ---
        for phrase in LOW_URGENCY_PHRASES:
            base = random.choice(templates)
            rows.append(make_row(base + " " + phrase, dept, "neutral", "low"))

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grievances.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records → {out_path}")
    print(f"Sentiment dist:\n{df['sentiment'].value_counts()}")
    print(f"Urgency dist:\n{df['urgency'].value_counts()}")
    return df


if __name__ == "__main__":
    generate()
