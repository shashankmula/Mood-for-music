from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# Load the trained model
loaded_model = load_model("ajmodel1.h5")

# Define class names for mood prediction
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Define a function to preprocess image for prediction
def preprocess_image(image):
    # Resize the image to match model input size
    resize_image = image.resize((128, 128))
    # Convert image to numpy array
    input_data = np.array(resize_image)
    # Normalize pixel values
    input_data = input_data / 255
    # Expand dimensions to match model input shape
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get image file from the request
        file = request.files['image']
        # Read the image using OpenCV
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Convert the image to PIL format
        image_fromarray = Image.fromarray(image, 'RGB')
        # Preprocess the image for prediction
        input_data = preprocess_image(image_fromarray)
        # Predict mood class
        pred = loaded_model.predict(input_data)
        result = class_names[np.argmax(pred)]
        return jsonify({'mood': result})

if __name__ == '__main__':
    app.run(debug=True)
