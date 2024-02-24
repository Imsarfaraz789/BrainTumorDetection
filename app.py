import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

model = load_model('BrainTumor10EpochsCategorical.keras')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img):
    image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_label = np.argmax(result, axis=1)[0]
    return class_label



@app.route('/', methods=['GET'])
def index():
    return "<p>Hello, World!</p>"

@app.route('/predict', methods=['POST'])
def upload():
     if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        if file:
            class_label = getResult(file)
            class_name = get_className(class_label)
            return jsonify({"result": class_name})
        return "No Prediction"

if __name__ == '__main__':
    app.run(debug=True)
