from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = "plants123"

# -----------------------------
# MODEL PATH  (Render server path)
# -----------------------------
MODEL_PATH = os.path.join("model", "model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("Model Loaded Successfully")

# -----------------------------
# UPLOAD FOLDER
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
LABEL_MAP_FILE = "label_map.json"

if not os.path.exists(LABEL_MAP_FILE):
    raise FileNotFoundError("label_map.json not found.")

with open(LABEL_MAP_FILE, "r") as f:
    class_names = json.load(f)

print("Class names loaded:", class_names)

# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # IMAGE PREPROCESSING
    target_size = model.input_shape[1:3]
    img = image.load_img(filepath, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    # PREDICT
    pred = model.predict(arr)
    cls = int(np.argmax(pred[0]))
    predicted_class = class_names[str(cls)]
    accuracy = round(float(np.max(pred[0]) * 100), 2)

    session['prediction_text'] = f"{predicted_class} ({accuracy}% confidence)"
    session['image_path'] = f"static/uploads/{file.filename}"

    return redirect(url_for('result'))

@app.route('/result')
def result():
    if "prediction_text" not in session:
        return redirect(url_for('dashboard'))

    return render_template(
        'result.html',
        prediction_text=session['prediction_text'],
        image_path=session['image_path']
    )

@app.route('/analyze_again')
def analyze_again():
    return render_template('home.html')

# -----------------------------
# RENDER DEPLOY PORT CONFIG
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
