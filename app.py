from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np

# Load the saved model
with open("best_model_rfc.pkl", "rb") as file:
    model = pickle.load(file)

# Define the input data schema using Pydantic
class StrokePredictionInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float

# Initialize FastAPI app
app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define a root endpoint to serve the HTML file
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return """
    <html>
        <head>
            <title>Stroke Prediction API</title>
        </head>
        <body>
            <h1>Welcome to the Stroke Prediction API!</h1>
            <form action="/predict" method="post">
                Age: <input type="text" name="age"><br>
                Hypertension: <input type="text" name="hypertension"><br>
                Heart Disease: <input type="text" name="heart_disease"><br>
                Avg Glucose Level: <input type="text" name="avg_glucose_level"><br>
                BMI: <input type="text" name="bmi"><br>
                Feature 6: <input type="text" name="feature_6"><br>
                Feature 7: <input type="text" name="feature_7"><br>
                Feature 8: <input type="text" name="feature_8"><br>
                Feature 9: <input type="text" name="feature_9"><br>
                Feature 10: <input type="text" name="feature_10"><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

# Define the prediction endpoint
@app.post("/predict")
def predict_stroke(age: float = Form(...), 
                  hypertension: int = Form(...), 
                  heart_disease: int = Form(...), 
                  avg_glucose_level: float = Form(...), 
                  bmi: float = Form(...),
                  feature_6: float = Form(...),
                  feature_7: float = Form(...),
                  feature_8: float = Form(...),
                  feature_9: float = Form(...),
                  feature_10: float = Form(...)):
    # Convert input data to a numpy array
    input_array = np.array([[
        age,
        hypertension,
        heart_disease,
        avg_glucose_level,
        bmi,
        feature_6,
        feature_7,
        feature_8,
        feature_9,
        feature_10
    ]])

    # Make a prediction
    prediction = model.predict(input_array)
    prediction_prob = model.predict_proba(input_array)

    # Return the prediction and probability
    return {
        "prediction": int(prediction[0]),
        "probability": float(prediction_prob[0][1])  # Probability of class 1 (stroke)
    }