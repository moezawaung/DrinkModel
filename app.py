from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = load_model("best_model_v2.keras")
labels = ['Coca-Cola', 'Heineken', 'Pepsi', ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file temporarily
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).resize((128, 128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    label = labels[class_index]

    return render_template('index.html', filename=file.filename, prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
