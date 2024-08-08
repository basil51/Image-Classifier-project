import tensorflow as tf
import numpy as np
from PIL import Image
import json

def process_image(image_path):
    """Preprocess the image to the required size and scale for the model."""
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  # Normalize to [0, 1]
    return image.numpy()

def load_model(model_path):
    """Load the saved Keras model."""
    #tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict(image_path, model, top_k):
    """Predict the top K classes for the given image using the trained model."""
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    top_k_probs = np.sort(predictions[0])[-top_k:][::-1]
    top_k_classes = np.argsort(predictions[0])[-top_k:][::-1]
    return top_k_probs, top_k_classes

def load_class_names(json_path):
    """Load class names from a JSON file."""
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names
