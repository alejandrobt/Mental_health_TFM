from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    scores: dict

app = FastAPI(
    title="NLP Prediction API",
    description="Servicio REST para inferencia de tu modelo BERT fine-tuneado",
    version="1.0"
)


# Carga del modelo entrenado
MODEL_NAME = "./results/checkpoint-240"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict_labels(text: str):
    # Tokenización y creación de tensores
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logs = model(**inputs).logits
    # Probabilidades softmax
    probs = torch.softmax(logs, dim=-1).squeeze().tolist()
    # Mapear ids a etiquetas
    id2label = model.config.id2label
    scores = {id2label[i]: probs[i] for i in range(len(probs))}
    # Seleccionar etiqueta con mayor probabilidad
    label = max(scores, key=scores.get)
    return label, scores

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")
    label, scores = predict_labels(request.text)
    return PredictionResponse(label=label, scores=scores)


#uvicorn --app-dir "C:\Users\carlo\OneDrive\Escritorio\Master\TFM" main:app --host 0.0.0.0 --port 8000 --reload
