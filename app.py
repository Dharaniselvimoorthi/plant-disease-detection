from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------- Flask app --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "plants123")

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# -------------------- Load model --------------------
model = load_model(MODEL_PATH)
print("Loaded model:", model.input_shape)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Class names --------------------
# Must match the order used during model training
class_names = [
    "Pepper_bell__healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -------------------- Routes --------------------
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if request.method == 'POST':
        return render_template('home.html')
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

    # Preprocess image
    target_size = model.input_shape[1:3]
    img = image.load_img(filepath, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    # Prediction
    pred = model.predict(arr)
    cls = np.argmax(pred[0])
    predicted_class = class_names[cls]
    accuracy = round(float(np.max(pred[0]) * 100), 2)

    # Save in session
    session['prediction_text'] = f"{predicted_class} ({accuracy}% confidence)"
    session['image_path'] = f"static/uploads/{file.filename}"
    return redirect(url_for('result'))

@app.route('/result')
def result():
    if "prediction_text" not in session:
        return redirect(url_for('dashboard'))
    return render_template('result.html',
                           prediction_text=session['prediction_text'],
                           image_path=session['image_path'])

# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
