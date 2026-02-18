import json
from xml.parsers import expat
import numpy as np
import joblib
import argparse
from pathlib import Path

model_path = Path('artifacts/model.pkl')
def load_model():
    if not model_path.exists():
        raise FileNotFoundError("Model file not found please train the model first")
    return joblib.load(model_path)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Feature list as JSON string. Example: \"[5.1,3.5,1.4,0.2]\"")
    args = parser.parse_args()

    #parse the input features
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError("Invalid input. Use JSON list, e.g. --input \"[5.1,3.5,1.4,0.2]\"")
    
    x = np.array(features).reshape(1, -1)
    model = load_model()
    prediction = model.predict(x)

    print(json.dumps({'prediction': prediction.tolist()}))
if __name__ == "__main__":
    main()