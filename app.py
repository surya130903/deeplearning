from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model
model = load_model('model_project.h5')

@app.route('/')
def home():
    return "Selamat Datang di API Prediksi"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pm10 = data['pm10']
    so2 = data['so2']
    co = data['co']
    o3 = data['o3']
    no2 = data['no2']
    max_val = data['max']

    input_query = np.array([[pm10, so2, co, o3, no2, max_val]])
    prediction = model.predict(input_query)
    predicted_class = np.argmax(prediction, axis=1)[0]

    categories = ['BAIK', 'SEDANG', 'TIDAK SEHAT']
    return jsonify({'kategori': categories[predicted_class]})

if __name__ == '__main__':
    app.run(debug=True)
