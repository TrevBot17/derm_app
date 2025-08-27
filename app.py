import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your model
model = load_model('my_model.keras')

st.title("BCC Detector")
st.write("Upload a skin lesion image for analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    result = "Yes_BCC" if probability > 0.5 else "No_BCC"
    confidence = probability if result == "Yes_BCC" else (1 - probability)
    
    # Display
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"**Prediction:** {result}")
    st.info(f"**Confidence:** {confidence:.2%}")