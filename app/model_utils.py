import cv2
import numpy as np
from keras.models import load_model

# Load your trained segmentation model
def load_segmentation_model():
    # Adjust the path to your trained model file
    model_path = 'JAN2024HACKATHON\model (1).h5'
    model = load_model(model_path)
    return model

# Preprocess the input image for segmentation
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Normalize pixel values to be in the range [0, 1]
    image = image / 255.0

    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)

    return image

# Process the image through the segmentation model
def process_image(image_path):
    # Load the segmentation model
    segmentation_model = load_segmentation_model()

    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Perform segmentation prediction
    segmentation_result = segmentation_model.predict(input_image)

    return segmentation_result
