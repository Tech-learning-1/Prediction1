from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import os
import json

app = Flask(__name__)
model_path = Path('artifacts/model.pkl')
if not model_path.exists():
    import train as trn
    trn.main()
model = joblib.load(model_path)

@app.route('/health', methods=['GET'])
def heatlth():
    return jsonify({'status': 'ok'})
@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    try:
       features = data['features']
       prediction = model.predict([features])
       return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
          return jsonify({'error': str(e)}), 400
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)