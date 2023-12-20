from flask import Flask, request
from werkzeug.utils import secure_filename
from plant_detection import single_prediction
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join('C:/Users/ganks/OneDrive/사진/plants/', filename)
    image_file.save(filepath)
    result = single_prediction(filepath)
    return {'prediction': result}