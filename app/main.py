import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from fpdf import FPDF
from urllib.parse import quote_plus
from googletrans import Translator

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Hugging Face API setup
HUGGINGFACE_API_TOKEN = st.secrets["hf"]["api_key"]
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Language map with font file names
language_map = {
    "English": ("en", "DejaVuSans.ttf"),
    "தமிழ்": ("ta", "NotoSansTamil.ttf"),
    "हिन्दी": ("hi", "NotoSansDevanagari.ttf"),
    "తెలుగు": ("te", "NotoSansTelugu.ttf"),
    "മലയാളം": ("ml", "NotoSansMalayalam.ttf")
}


# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Prediction
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_index)]


# AI explanation
def get_disease_explanation(disease_name):
    prompt = (
        f"You are a plant disease expert. A farmer is seeing symptoms of '{disease_name}' on their crops.\n\n"
        f"Please provide detailed information in the following format:\n\n"
        f"🤠 Cause of the disease:\n"
        f"🧪 Nutrient deficiencies involved (if any):\n"
        f"🌿 Organic treatment options:\n"
        f"💊 Chemical treatment options:\n"
        f"🛡️ Prevention tips for the future:\n"
    )
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()[0]["generated_text"]
        first_bullet_index = output.find("🤠")
        return output[first_bullet_index:].strip() if first_bullet_index != -1 else output.strip()
    return f"❌ Error fetching AI response: {response.status_code} - {response.text}"


# PDF generation
def generate_pdf(prediction, advice_text, language_code, font_filename):
    translator = Translator()
    if language_code != 'en':
        prediction = translator.translate(prediction, dest=language_code).text
        advice_text = translator.translate(advice_text, dest=language_code).text

    font_path = os.path.join(working_dir, "fonts", font_filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("CustomFont", "", font_path, uni=True)
    pdf.set_font("CustomFont", size=12)
    pdf.multi_cell(0, 10, f"\U0001f33e Predicted Disease: {prediction}\n\n{advice_text}")

    pdf_path = os.path.join(working_dir, "ai_advice.pdf")
    pdf.output(pdf_path)
    return pdf_path, prediction, advice_text


# Streamlit UI
st.set_page_config(page_title="🌿 Smart Crop Disease Assistant", layout="centered")
st.title('🍃 Crop Disease Detection & AI Cure Advisor')

# Language selection
language_choice = st.selectbox("🌐 Select your preferred language", list(language_map.keys()))
selected_lang_code, font_file = language_map[language_choice]

uploaded_image = st.file_uploader("📸 Upload a crop leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    ai_explanation = None
    prediction = None

    with col2:
        if st.button('🔍 Diagnose Disease'):
            with st.spinner("🧠 Identifying disease..."):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'🧬 **Predicted Disease:** `{prediction}`')

            with st.spinner("📡 Consulting AI Expert..."):
                ai_explanation = get_disease_explanation(prediction)

    if ai_explanation:
        st.markdown("---")
        st.markdown("### 🤖 AI-Generated Cure & Advice")

        # Translate for display if needed
        if selected_lang_code != 'en':
            translator = Translator()
            ai_explanation = translator.translate(ai_explanation, dest=selected_lang_code).text
            prediction = translator.translate(prediction, dest=selected_lang_code).text

        st.markdown(ai_explanation)

        # Generate & Download PDF
        pdf_path, translated_pred, translated_adv = generate_pdf(
            prediction, ai_explanation, selected_lang_code, font_file
        )
        with open(pdf_path, "rb") as file:
            st.download_button("📄 Download as PDF", file, file_name="plant_disease_advice.pdf")

        # WhatsApp Share
        message = f"🌾 Plant Disease Info\n\n🧬 Prediction: {translated_pred}\n\n{translated_adv}"
        encoded_message = quote_plus(message)
        whatsapp_url = f"https://wa.me/?text={encoded_message}"
        st.markdown(f"[📱 Share on WhatsApp]({whatsapp_url})", unsafe_allow_html=True)
