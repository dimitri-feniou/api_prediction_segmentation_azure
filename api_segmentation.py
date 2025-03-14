from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from pathlib import Path
from tensorflow import keras
import os
import gdown
import matplotlib.pyplot as plt
import cv2
from fastapi.responses import Response

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
    
    print(f"🔎 Valeurs min/max de l’image: {image.min()} - {image.max()}")  # 🔥 Debug
    
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
    return image



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint de prédiction"""
    image = await file.read()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Obtenir les classes
    mask = np.argmax(prediction, axis=-1)[0]
    
    # Debugging : afficher les valeurs uniques et leur distribution
    unique_values = np.unique(mask)
    print(f"Valeurs uniques dans le masque: {unique_values}")
    print(f"Nombre de pixels par classe: {[np.sum(mask == val) for val in unique_values]}")
    
    # Si le masque n'a qu'une seule valeur, essayez d'utiliser les probabilités brutes
    if len(unique_values) <= 1:
        print("⚠️ Masque uniforme détecté ! Utilisation des probabilités brutes.")
        # Prendre le canal avec la plus haute probabilité pour chaque pixel
        mask = prediction[0, :, :, 0] * 255  # Supposons que le premier canal est significatif
    
    # Normaliser pour s'assurer que les valeurs sont visibles
    # Étirer l'histogramme pour utiliser toute la plage 0-255
    if mask.max() > mask.min():  # Éviter la division par zéro
        mask_normalized = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
    else:
        mask_normalized = np.zeros_like(mask, dtype=np.uint8)
    
    # Appliquer une colormap pour rendre les différences plus visibles
    # Convertir d'abord en image à 3 canaux RGB
    colored_mask = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
    
    # Conversion en image
    _, buffer = cv2.imencode(".png", colored_mask)
    
    return Response(content=buffer.tobytes(), media_type="image/png")



