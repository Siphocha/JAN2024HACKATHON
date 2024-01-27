import cv2
import numpy as np
import os
import keras
from keras.models import load_model
from tempfile import NamedTemporaryFile

def iou_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(np.abs(y_true * y_pred), axis=[1, 2, 3])
    union = np.sum(y_true, axis=[1, 2, 3]) + np.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = np.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# Load your trained segmentation model
def load_segmentation_model():
    try:
        current_directory = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_directory, 'model (1).h5')

        with keras.utils.custom_object_scope({'iou_coef': iou_coef}):
            model = load_model(model_path)

        print("Model loaded successfully.")
        return model

    except Exception as e:
        print("Error loading the model:", str(e))
        return None

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

    # Save the segmented image to a temporary file
    with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        # Assuming segmentation_result is an image, adjust the saving accordingly
        cv2.imwrite(temp_file.name, segmentation_result[0] * 255)

    return temp_file.name
