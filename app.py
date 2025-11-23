from flask import Flask, render_template, request
import numpy as np
import pickle
import datetime as dt

app = Flask(__name__)

# Load model + label encoder
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from UI
        date_str = request.form["date"]
        precipitation = float(request.form["precipitation"])
        temp_max = float(request.form["temp_max"])
        temp_min = float(request.form["temp_min"])
        wind = float(request.form["wind"])

        # Convert date to ordinal
        date_obj = dt.datetime.strptime(date_str, "%Y-%m-%d")
        date_ordinal = date_obj.toordinal()

        # Create feature array
        features = np.array([[date_ordinal, precipitation, temp_max, temp_min, wind]])

        # Predict encoded label
        pred_encoded = model.predict(features)[0]

        # Decode weather type
        prediction = le.inverse_transform([pred_encoded])[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
