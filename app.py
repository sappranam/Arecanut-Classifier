from flask import Flask, request, render_template, send_from_directory, url_for
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('arecanut_model.pkl')
IMAGE_SIZE = (64, 64)

# Upload folder setup
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Proper image read (fixing your earlier bug)
        img = cv2.imread(file_path)
        img_resized = cv2.resize(img, IMAGE_SIZE).flatten()
        prediction = model.predict([img_resized])[0]
        prediction_prob = model.predict_proba([img_resized])[0]

        # Label Mapping
        labels = ['High', 'Medium', 'Low']
        result = labels[prediction]
        confidence = max(prediction_prob) * 100

        return render_template('result.html', result=result, confidence=round(confidence, 2), filename=filename)

    except Exception as e:
        return f"Error processing file: {e}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
