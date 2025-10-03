import streamlit as st
import joblib
import cv2  # ✅ สำคัญ! ต้อง import
from PIL import Image
import numpy as np

# --- Load the trained model ---
with open("svm_image_classifier_model.pkl", "rb") as f:
    model = joblib.load(f)

# --- Streamlit UI ---
st.title("Fruit Classifier")
st.write("Upload an image of an **apple** or **orange**, and the model will predict it.")

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# --- Define class labels ---
class_dict = {0: "Apple", 1: "Orange"}

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        # ✅ Resize image to 100x100 (เหมือนตอนฝึกโมเดล)
        image = image.resize((100, 100))

        # ✅ Convert to numpy array
        image_array = np.array(image)

        # ✅ Convert RGB to BGR (ตามลำดับสีของ OpenCV)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # ✅ Flatten และ reshape เป็น input ที่โมเดลต้องการ
        image_array = image_array.flatten().reshape(1, -1)

        # ✅ Predict
        prediction = model.predict(image_array)[0]
        prediction_name = class_dict[prediction]

        # ✅ Show result
        st.markdown(f"### Prediction: **{prediction_name}**")
