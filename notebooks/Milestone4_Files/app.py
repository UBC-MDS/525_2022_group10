from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 1. Load your model here
model = joblib.load("model.joblib")

# 2. Define a prediction function
def return_prediction(input_data):

    # format input_data here so that you can pass it to model.predict()
    reformatted = np.array(input_data).reshape(1, -1)
    return model.predict(reformatted)


# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """


# 4. define a new route which will accept POST requests and return model predictions
@app.route("/predict", methods=["POST"])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    inputs = content["data"]
    prediction = return_prediction(inputs)
    results = {
        "Input": inputs,
        "Output": f"Here is the prediction based on our best performing ensemble model: {prediction}!",
    }
    # a dictionary of things we want to return
    return jsonify(results)
