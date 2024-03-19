
import streamlit as st
import os
import json
from streamlit.logger import get_logger
from gradio_client import Client
import math
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

LOGGER = get_logger(__name__)

# Client de l'API 
client = Client("https://abdoulgafar-woodsound.hf.space/")

# stocker le fichier 
def save_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Fonction d'envoie de mail
def send_email(receiver_email, subject, message):
    load_dotenv() # charger les secrets de .env
    sender_email = os.getenv("GMAIL_ADDRESS")  
    sender_password = os.getenv("GMAIL_PASSWORD")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        st.success("E-mail envoyé avec succès!")
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de l'e-mail: {e}")
    finally:
        server.quit()


def run():
    st.set_page_config(
        page_title="Application",
        page_icon="👋",
    )
    st.title("👋 Détection de son de coupe de bois")

    uploaded_file = st.file_uploader("Sélectionnez ou enregistrer un fichier audio (wav, mp3, etc.)")

    # Charger le fichier, consommer l'api, et recupérer le retour
    if uploaded_file is not None:
        st.subheader("Extrait audio")
        st.audio(uploaded_file)
        result = client.predict(save_file(uploaded_file), api_name="/predict")
        # st.write("Résultat de la prédiction :", result)       
        with open(result[1], 'r') as f:
            data = json.load(f)
        predictions = data["confidences"]

        # predictions et labels
        st.subheader("Prédictions")
        for pred in predictions:
            col1, col2, col3 = st.columns(3)
            col1.write(pred['label'])
            col2.progress(pred['confidence'])
            col3.info(math.trunc(pred['confidence'] * 100) / 100)

        st.subheader("Spectrogramme")
        st.image(result[0]) 

        # envoie de notification mail
        first_prediction = predictions[0]
        if (first_prediction["label"] == "scie électrique" ) and (first_prediction["confidence"] > 0.7) :
            st.success("Envoie d'une notification en cours ...")
            # charger secret
            load_dotenv()
            receiver_email = os.getenv("GMAIL_RECEIVER")
            subject = "Détection de son de coupe de bois"
            message = f"\n {first_prediction['label']} \n {first_prediction['confidence']}"

            # send_email(receiver_email, subject, message)
        

      
        


if __name__ == "__main__":
    run()
