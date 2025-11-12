#!/usr/bin/env python3
"""
Simple Flask server to expose the Iris classifier as an HTTP API.
"""
from flask import Flask, request, jsonify
import os
import traceback
import numpy as np

from model import IrisClassifier
from data_loader import load_iris_data, get_target_names

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/iris_classifier.pkl")


def get_classifier():
    clf = IrisClassifier()
    try:
        clf.load_model(MODEL_PATH)
    except Exception:
        # If the model is not present, train a fresh one and save it
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=0)
        clf.train(X_train, y_train)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        clf.save_model(MODEL_PATH)
    return clf


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        # Accept either a JSON array or {"instances": [...]} payload
        if isinstance(payload, dict) and "instances" in payload:
            instances = payload["instances"]
        else:
            instances = payload

        arr = np.array(instances)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        clf = get_classifier()
        preds = clf.predict(arr).tolist()
        target_names = get_target_names()
        labels = [target_names[p] for p in preds]

        return jsonify(predictions=preds, labels=labels)

    except Exception as e:
        return jsonify(error=str(e), tb=traceback.format_exc()), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

