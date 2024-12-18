import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import torch
from preprocessing import xception_preprocess, densenet_preprocess, resnet_preprocess


# Load pre-trained models
@st.cache_resource
def load_trained_models():
    models = {
        'Xception': load_model(r'C:\Users\MBR\Downloads\deeeeep\final_xception_finetuned_model.keras'),
        'DenseNet': load_model(r'C:\Users\MBR\Downloads\deeeeep\Densnet.keras'),
        # 'ResNet50': load_model(r'C:\Users\MBR\Downloads\deeeeep\trained_model (1).pth', map_location=torch.device('cpu'))
    }
    return models

# Preprocess uploaded image for the model
def preprocess_image(img, model_name):
    if model_name == 'Xception':
        return xception_preprocess(img)
    elif model_name == 'DenseNet':
        return densenet_preprocess(img)
    else:  # ResNet50
        return resnet_preprocess(img)

# Get predictions from the model
def get_top_predictions(model, preprocessed_img, top_k=1):
    predictions = model.predict(preprocessed_img)
    class_indices = {str(i): f"Class {i}" for i in range(predictions.shape[1])}  # Replace with your labels
    sorted_indices = predictions[0].argsort()[::-1][:top_k]

    results = [
        (class_indices[str(i)], predictions[0][i]) 
        for i in sorted_indices
    ]
    return results

# Main application
def main():
    st.title("Face Classification")
    st.write("Classify faces using pre-trained deep learning models.")
    
    # Load models
    models = load_trained_models()

    # Sidebar: Model selection
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose a Model",
        list(models.keys())
    )

    # Image upload
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or select an image file (JPG/PNG)", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        img = image.load_img(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        try:
            # Get the selected model
            selected_model = models[selected_model_name]

            # Preprocess the image
            preprocessed_img = preprocess_image(img, selected_model_name)

            # Get predictions
            st.subheader(f"Classification Results - {selected_model_name}")
            top_predictions = get_top_predictions(selected_model, preprocessed_img)

            # Display predictions
            for i, (class_name, probability) in enumerate(top_predictions, 1):
                st.write(f"**{i}. {class_name}**: {probability:.2%}")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    main()
