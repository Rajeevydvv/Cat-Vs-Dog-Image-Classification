import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Title
st.title("🐱 Cat vs Dog Image Classifier")
st.write("Upload an image and the model will predict whether it is a Cat or Dog.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image, verbose=0)[0][0]

    st.subheader("Prediction")

    if prediction > 0.5:
        st.success(f"🐶 Dog ({prediction:.2%})")
    else:
        st.success(f"🐱 Cat ({(1-prediction):.2%})")

    st.progress(float(max(prediction, 1-prediction)))

    st.write(f"Raw Model Output: {prediction:.4f}")