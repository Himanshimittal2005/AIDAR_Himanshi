import streamlit as st
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from keras.models import load_model
from transformers import pipeline
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
from gtts import gTTS
from langdetect import detect
import os
import tempfile


# === PAGE CONFIG ===
st.set_page_config(page_title="AIDAR – AI Disaster Awareness", layout="centered")
st.title("\U0001F310 AIDAR – AI Disaster Awareness & Response System")

st.markdown("This app combines **image classification**, **tweet analysis**, and **risk score prediction** to support early disaster response and awareness.")

# === LOAD MODELS ===
@st.cache_resource
def load_all_models():
    image_model = load_model("disaster_classifier_model.h5")
    risk_model = joblib.load("risk_score_model2.pkl")
    model_columns = joblib.load("model_columns2.pkl")
    tweet_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return image_model, risk_model, model_columns, tweet_classifier

image_model, risk_model, model_columns, tweet_classifier = load_all_models()

# === TABS FOR THREE MODULES ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["\U0001F4F7 Image Classifier", "🥆 Tweet Classifier", "📊 Risk Score Predictor", "\U0001F5FA️ Risk Heatmap", "\U0001F5E3️ Chatbot"])

# === MODULE 1: Image Classification ===
with tab1:
    st.subheader("Upload a Disaster Image")
    uploaded_file = st.file_uploader("Upload image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_tensor)
        predicted_class = np.argmax(prediction)

        class_names = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
        st.success(f"\U0001F9E0 Predicted Disaster: **{class_names[predicted_class]}**")

# === MODULE 2: Tweet Classification ===
with tab2:
    st.subheader("Analyze a Tweet for Emergency Detection")
    user_tweet = st.text_area("Enter or paste a tweet:")
    if st.button("Analyze Tweet"):
        if user_tweet.strip() != "":
            result = tweet_classifier(user_tweet)[0]
            label = result['label']
            score = round(result['score'] * 100, 2)
            status = "🚨 Emergency" if label == "NEGATIVE" else "✅ Non-Emergency"

            st.info(f"**Predicted Sentiment:** {label} ({score}%)")
            st.success(f"**Tweet Interpretation:** {status}")
        else:
            st.warning("Please enter a tweet.")

# === MODULE 3: Risk Score Predictor ===
with tab3:
    st.subheader("Disaster Risk Score Predictor")

    all_states = [col.replace("State_", "") for col in model_columns if col.startswith("State_")]
    selected_state = st.selectbox("Select State", sorted(all_states))

    avg_temp_1 = st.slider("Average Temperature (°C)", 0.0, 60.0, 30.0, key="avg_temp_1")
    avg_humidity_1 = st.slider("Average Humidity (%)", 0.0, 100.0, 70.0, key="avg_humidity_1")
    wind_speed_1 = st.slider("Wind Speed (km/h)", 0.0, 150.0, 10.0, key="wind_speed_1")

    input_dict = {
        'Avg_Temp': avg_temp_1,
        'Avg_Humidity': avg_humidity_1,
        'Wind_Speed': wind_speed_1
    }

    for col in model_columns:
        if col.startswith("State_"):
            input_dict[col] = 1 if col == f"State_{selected_state}" else 0
        elif col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])[model_columns]

    if st.button("Predict Risk Score"):
        score = risk_model.predict(input_df)[0]

        if score < 3:
            risk_level = "\U0001F7E2 Low"
        elif score < 7:
            risk_level = "\U0001F7E0 Medium"
        else:
            risk_level = "\U0001F534 High"

        st.metric("Predicted Risk Score (0–10)", round(score, 2))
        st.success(f"Estimated Risk Level: **{risk_level}**")

# === MODULE 4: Risk Heatmap ===
with tab4:
    st.subheader("\U0001F5FA️ State-wise Predicted Risk Heatmap")

    with open("india_states.geojson", "r") as f:
        geojson_data = json.load(f)

    st.markdown("Customize the weather parameters to update the heatmap dynamically:")

    avg_temp_2 = st.slider("Average Temperature (°C)", 0.0, 60.0, 30.0, key="avg_temp_2")
    avg_humidity_2 = st.slider("Average Humidity (%)", 0.0, 100.0, 70.0, key="avg_humidity_2")
    wind_speed_2 = st.slider("Wind Speed (km/h)", 0.0, 150.0, 10.0, key="wind_speed_2")

    all_states_map = [col.replace("State_", "") for col in model_columns if col.startswith("State_")]

    state_risk_scores = {}
    for state in all_states_map:
        input_dict = {
            'Avg_Temp': avg_temp_2,
            'Avg_Humidity': avg_humidity_2,
            'Wind_Speed': wind_speed_2
        }
        for col in model_columns:
            if col.startswith("State_"):
                input_dict[col] = 1 if col == f"State_{state}" else 0
            elif col not in input_dict:
                input_dict[col] = 0

        input_df = pd.DataFrame([input_dict])[model_columns]
        predicted_score = risk_model.predict(input_df)[0]
        state_risk_scores[state] = round(predicted_score, 2)

    risk_df = pd.DataFrame(list(state_risk_scores.items()), columns=["State", "Risk"])

    m = folium.Map(location=[23.5937, 80.9629], zoom_start=4)

    folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=risk_df,
        columns=["State", "Risk"],
        key_on="feature.properties.NAME_1",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Predicted Risk Score (0–10)",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=700, height=500)

# === MODULE 5: Chatbot ===
@st.cache_resource
def load_chatbot():
    return pipeline(
        "question-answering",
        model="bert-base-multilingual-cased",
        tokenizer="bert-base-multilingual-cased"
    )

qa_pipeline = load_chatbot()

DISASTER_FAQ_CONTEXT = """
During a flood, move to higher ground. Avoid walking or driving through floodwaters. Disconnect electrical appliances. Listen to local authorities for instructions.
In a wildfire, evacuate immediately if told to do so. Keep emergency supplies ready. Close windows to prevent smoke from entering.
During an earthquake, drop, cover, and hold on. Stay indoors away from windows. After shaking stops, evacuate carefully.
In a cyclone, secure your home, stock essentials, and stay away from windows. Follow evacuation orders if given.
"""

with tab5:
    st.subheader("🗣️ AIDAR – Multilingual Disaster Assistant")

    user_input = st.text_input("Ask a disaster-related question (Hindi/English/Bengali):")

    if st.button("Ask AIDAR"):
        if user_input.strip():
            language = detect(user_input)
            result = qa_pipeline({
                'question': user_input,
                'context': DISASTER_FAQ_CONTEXT
            })
            answer = result['answer']

            st.markdown(f"🌍 Detected Language: **{language}**")
            st.markdown(f"🤖 AIDAR says: **{answer}**")

            try:
                tts_lang = language if language in ['en', 'hi', 'bn'] else 'en'
                tts = gTTS(answer, lang=tts_lang)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format="audio/mp3")
            except Exception as e:
                st.error(f"Audio generation failed: {e}")
        else:
            st.warning("Please enter a question.")