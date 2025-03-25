import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress Keras warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
image_model = load_model('skin_disease_model.h5')

# Class labels
class_labels = ['Acne','Actinic_Keratosis','Benign_tumors','Bullous','Candidiasis','DrugEruption','Eczema','Infestations_Bites','Lichen','Lupus','Moles','Psoriasis','Rosacea','Seborrh_Keratoses','SkinCancer','Sun_Sunlight_Damage','Tinea','Unknown_Normal','Vascular_Tumors','Vascultis','Vitiligo','Warts']  # Replace with your actual class labels

def predict_disease_from_image(img_path):
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class
    prediction = image_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return class_labels[predicted_class[0]]

# Example usage
img_path = r'G:\Chatbot\DERMNET\test\Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions\actinic-cheilitis-sq-cell-lip-6.jpg'  # Replace with the path to your test image
predicted_disease = predict_disease_from_image(img_path)
print(f"Predicted Disease: {predicted_disease}") 