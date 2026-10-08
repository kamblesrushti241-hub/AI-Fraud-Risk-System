from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained AI model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    amount = float(request.form["amount"])
    balance = float(request.form["balance"])
    previous_transactions = int(request.form["previous_transactions"])
    international = int(request.form["international"])
    unusual_location = int(request.form["unusual_location"])

    # Prepare input for AI model
    features = [[
        amount,
        balance,
        previous_transactions,
        international,
        unusual_location
    ]]

    # AI prediction
    prediction = model.predict(features)[0]

    # Fraud probability
    probability = model.predict_proba(features)[0][1] * 100

    # Financial risk score
    risk_score = round(probability, 2)

    # Risk level and recommendation
    if risk_score >= 70:
        risk_level = "HIGH RISK"
        recommendation = "Block or manually review this transaction."

    elif risk_score >= 40:
        risk_level = "MEDIUM RISK"
        recommendation = "Verify the transaction before approving it."

    else:
        risk_level = "LOW RISK"
        recommendation = "Transaction appears safe to proceed."

    # Transaction result
    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Genuine Transaction"

    return render_template(
        "result.html",
        result=result,
        probability=round(probability, 2),
        risk_score=risk_score,
        risk_level=risk_level,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)