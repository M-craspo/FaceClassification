from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

def xception_preprocess(uploaded_image):
    """
    Preprocess the uploaded image for the Xception model.
    Input:
        uploaded_image: PIL Image uploaded in Streamlit
    Returns:
        Preprocessed image ready for Xception model
    """
    # Resize to 299x299 (Xception input size)
    image = uploaded_image.resize((299, 299))
    
    # Convert to array and normalize using preprocess_input
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # Handle RGBA images
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    preprocessed_image = xception_preprocess_input(image_array)  # Apply Xception preprocessing

    return preprocessed_image


def densenet_preprocess(uploaded_image):
    """
    Preprocess the uploaded image for the DenseNet121 model.
    Input:
        uploaded_image: PIL Image uploaded in Streamlit
    Returns:
        Preprocessed image ready for DenseNet121 model
    """
    # Resize to 299x299 (DenseNet input size for your training setup)
    image = uploaded_image.resize((299, 299))
    
    # Convert to array and normalize using preprocess_input
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # Handle RGBA images
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    preprocessed_image = densenet_preprocess_input(image_array)  # Apply DenseNet preprocessing

    return preprocessed_image


def resnet_preprocess(uploaded_image):
    """
    Preprocess the uploaded image for the ResNet50 model.
    Input:
        uploaded_image: PIL Image uploaded in Streamlit
    Returns:
        Preprocessed image ready for ResNet50 model
    """
    # Define transforms for resizing, converting to tensor, and normalizing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # Convert PIL image to tensor and apply transformations
    image_tensor = transform(uploaded_image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
