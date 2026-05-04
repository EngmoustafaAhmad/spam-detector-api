from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re

app = FastAPI()

# Load the vectorizer (shared across models)
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load all 5 models
models = {
    "lr": pickle.load(open("lr.pkl", "rb")),
    "dtc": pickle.load(open("dtc.pkl", "rb")),
    "nb": pickle.load(open("nb.pkl", "rb")),
    "rfc": pickle.load(open("rfc.pkl", "rb")),
    "svc": pickle.load(open("svc.pkl", "rb"))
}

class EmailData(BaseModel):
    text: str
    algorithm: str  # e.g., "lr", "rfc", "svc"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/predict")
async def predict_spam(data: EmailData):
    # 1. Clean and Vectorize
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    
    # 2. Select Model
    model = models.get(data.algorithm, models["lr"]) # Default to Logistic Regression
    
    # 3. Predict
    prediction = model.predict(vectorized)
    result = "Spam" if prediction[0] == 1 else "Ham"
    
    return {
        "algorithm_used": data.algorithm,
        "prediction": result,
        "label": int(prediction[0])
    }
