from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import gdown
import zipfile

# Ruta donde estará el modelo
checkpoint_dir = "./results/checkpoint-240"
zip_path = "./results/model.zip"

# ID del archivo en Google Drive
file_id = "15gH7j7EKHgGozEvwV2IdAg_pi5eqlawu"
gdown_url = f"https://drive.google.com/uc?id={file_id}"

# Descargar y descomprimir si no existe
if not os.path.exists(checkpoint_dir):
    os.makedirs("./results", exist_ok=True)
    print("Descargando checkpoint desde Google Drive...")
    gdown.download(gdown_url, zip_path, quiet=False)

    print("Descomprimiendo modelo...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("./results")
else:
    print("Checkpoint ya existe.")

# FastAPI setup
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

# Cargar el modelo
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
model.eval()

def predict_labels(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    id2label = model.config.id2label
    scores = {id2label[i]: probs[i] for i in range(len(probs))}
    label = max(scores, key=scores.get)
    return label, scores

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")
    label, scores = predict_labels(request.text)
    return PredictionResponse(label=label, scores=scores)



#uvicorn --app-dir "C:\Users\carlo\OneDrive\Escritorio\Master\TFM" main:app --host 0.0.0.0 --port 8000 --reload
