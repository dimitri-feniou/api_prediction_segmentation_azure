from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from pathlib import Path
from tensorflow import keras
import os
import gdown



# ID du fichier Google Drive (remplacez par votre propre ID)
FILE_ID = "12vreJSXI9P0lpwF0gWnoKuG0T6RJnVFK"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "unet_model.h5")

def download_model_from_drive():
    """Télécharge le modèle depuis Google Drive si non présent localement."""
    # 🔹 Créer le dossier `model/` s'il n'existe pas
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"📁 Dossier créé : {MODEL_DIR}")

    if not os.path.exists(MODEL_PATH):
        print("🔄 Téléchargement du modèle depuis Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        print("✅ Modèle téléchargé avec succès !")
    else:
        print("✅ Modèle déjà présent localement.")

# Exécuter le téléchargement au démarrage
download_model_from_drive()

# Charger le modèle U-Net 
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "unet_model.h5"
model = load_model(MODEL_PATH)


app = FastAPI()

# Charger le modèle de segmentation
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé avec succès !")
def preprocess_image(image_bytes):
    """Prétraitement de l'image avant prédiction"""
    image = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    image = np.array(image) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint de prédiction"""
    image = await file.read()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    mask = np.argmax(prediction, axis=-1)[0]  # Obtenir les classes

    return {"mask": mask.tolist()}
