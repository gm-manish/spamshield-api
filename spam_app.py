from flask import Flask, request, jsonify
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = clean_text(data.get("message", ""))
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return jsonify({
        "prediction": "spam" if pred == 1 else "not spam",
        "confidence": round(float(prob), 3)
    })

if __name__ == "__main__":
    app.run()
