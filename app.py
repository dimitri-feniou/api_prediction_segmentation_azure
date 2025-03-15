import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import cv2

API_URL = "http://20.103.18.183:8000/predict/"

def upload_and_predict(image):
    """Envoie l'image à l'API et récupère le masque prédit"""
    # Resize image to 256x256 to match model input
    resized_image = image.resize((256, 256))
    
    img_bytes = io.BytesIO()
    resized_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("image.png", img_bytes, "image/png")}
    
    try:
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            # Traiter la réponse comme une image
            mask_image = Image.open(io.BytesIO(response.content))
            return mask_image, resized_image
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None, resized_image
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return None, resized_image

# Interface Streamlit
st.title("Application de Segmentation d'Images")

uploaded_file = st.file_uploader("Téléchargez une image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_column_width=True)

    if st.button("Lancer la prédiction"):
        with st.spinner("Traitement en cours..."):
            mask, resized_image = upload_and_predict(image)
            
            if mask is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption="Image redimensionnée", use_column_width=True)
                with col2:
                    st.image(mask, caption="Masque prédit", use_column_width=True)
                
                # Option: Ajouter une superposition du masque sur l'image
                try:
                    # Convertir les images en tableaux numpy
                    img_array = np.array(resized_image)
                    mask_array = np.array(mask)
                    
                    # Redimensionner le masque si nécessaire
                    if img_array.shape[:2] != mask_array.shape[:2]:
                        mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
                    
                    # Créer une superposition
                    alpha = 0.6  # Transparence du masque
                    overlay = cv2.addWeighted(img_array, 1-alpha, mask_array, alpha, 0)
                    
                    st.image(overlay, caption="Superposition masque/image", use_column_width=True)
                except Exception as e:
                    st.warning(f"Impossible de créer la superposition: {str(e)}")