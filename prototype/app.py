from flask import Flask, request
from onnxruntime import InferenceSession
import numpy as np
import pandas as pd
from skl2onnx.common.data_types import FloatTensorType
from joblib import load

app = Flask(__name__)
encoder = load("label_encoder.joblib")

@app.route("/", methods=['POST'])
def classify():
    data = request.json

    alcohol = data["alcohol"]
    malic_acid = data["malic_acid"]
    total_phenols = data["total_phenols"]
    flavanoids = data["flavanoids"]
    proanthocyanins = data["proanthocyanins"]
    color_intensity = data["color_intensity"]
    hue = data["hue"]
    od280_od315_of_diluted_wines = data["od280_od315_of_diluted_wines"]

    with open("wine_k-nN.onnx", "rb") as f:
        onx = f.read()
    sess = InferenceSession(onx, providers=["CPUExecutionProvider"])

    
    inp = np.array([[alcohol, malic_acid, total_phenols, flavanoids, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines]], dtype=np.float32)

    input_name = sess.get_inputs()[0].name  # "input" if you used that name

    prediction = sess.run(None, {input_name: inp})

    pred_labels = encoder.inverse_transform(prediction[0].astype(int))

    result = pd.DataFrame()
    result["class"] = pred_labels
    result["Probability"] = prediction[0]
    return {"prediction": result.to_json(orient="records")}

if __name__ == '__main__':
    app.run(debug=True)