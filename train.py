from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import pandas as pd
import json


def main():
    #load the data
    iris = load_iris()
    #split the data into training and testing sets
    X , Y  = iris.data, iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)

    # Save the model
    os.makedirs('artifacts', exist_ok=True)
    model_path = os.path.join('artifacts', 'model.pkl')
    joblib.dump(model, model_path)

    # Save the metrics
    accuracy = model.score(X_test, Y_test)
    metrics = {'accuracy': accuracy}
    metrics_path = os.path.join('artifacts', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f'Model trained and saved to {model_path}')
    print("accuracy: ", accuracy)
if __name__ == "__main__":
    main()
