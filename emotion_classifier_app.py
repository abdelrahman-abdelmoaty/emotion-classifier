import streamlit as st
import joblib
import numpy as np
import time

# Set page config
st.set_page_config(page_title="Emotion Detector", layout="centered")

# Cache model loading
@st.cache_resource
def load_model():
    try:
        model = joblib.load('emotion_classifier_model.joblib')
        if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
            return None, f"Loaded model ({type(model).__name__}) does not support predict or predict_proba methods."
        return model, None
    except FileNotFoundError:
        return None, "Model file 'emotion_classifier_model.joblib' not found."
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

# Load the model
pipeline, model_error = load_model()

# Emotion prediction function
def predict_emotion(text):
    if not pipeline:
        return None, None, "Model not loaded."

    emotion_map = {
        0: ('sadness', '😢'),
        1: ('joy', '😊'),
        2: ('love', '❤️'),
        3: ('anger', '😠'),
        4: ('fear', '😨'),
        5: ('surprise', '😯')
    }

    if not text or len(text.strip()) == 0:
        return None, None, "Please enter some text."
    if len(text.strip()) < 5:
        return None, None, "Warning: Very short text may lead to less accurate predictions. Proceed anyway?"

    try:
        prediction = pipeline.predict([text])[0]
        probabilities = pipeline.predict_proba([text])[0]
        confidence = np.max(probabilities) * 100
        return emotion_map[prediction], confidence, None
    except Exception as e:
        return None, None, f"Prediction failed: {str(e)}"

# Check for model loading errors
if model_error:
    st.error(model_error)
    st.stop()

# Title and description
st.title("Emotion Detector")
st.write("Discover the emotions behind your words")

# Pre-prompted text examples
sample_texts = {
    "Select a sample text": "",
    "Joyful moment": "I just got promoted at work, and I'm thrilled!",
    "Sad memory": "I miss my old dog who passed away last year.",
    "Angry reaction": "I'm so frustrated with this terrible customer service!",
}

# Dropdown for sample texts
selected_sample = st.selectbox("Choose a sample text or type your own", list(sample_texts.keys()))
default_text = sample_texts[selected_sample]

# Text input
user_input = st.text_area('', value=default_text, height=150, placeholder="Type your text here...")

# Analyze button
if st.button('Analyze Emotion'):
    if not user_input:
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing emotion..."):
            time.sleep(0.5)  # Simulate processing
            result, confidence, error = predict_emotion(user_input)

            if error:
                if "Warning" in error:
                    st.warning(error)
                else:
                    st.error(error)
            elif result:
                emotion, emoji = result
                st.subheader(f"{emotion.upper()} {emoji}")
                st.write(f"Confidence: {confidence:.1f}%")