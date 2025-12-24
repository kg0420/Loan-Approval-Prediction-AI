import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and encoders
model = joblib.load('Loan_Approval.pkl')
education_encoder = joblib.load('education_encoder.pkl')
self_employed_encoder = joblib.load('self_employed_encoder.pkl')


@app.route("/", methods=["GET", "POST"])
def loan_approval():
    return render_template("Loan_approval.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        education = request.form['education']
        self_employed = request.form['self_employed']
        income = float(request.form['income_annum'])
        loan_term = float(request.form['loan_term'])
        cibil_score = float(request.form['cibil_score'])
        loan_amount = float(request.form['loan_amount'])
        residential_assets_value = float(request.form['residential_assets_value'])
        commercial_assets_value = float(request.form['commercial_assets_value'])

        # Transform categorical inputs
        if education not in education_encoder.classes_ or self_employed not in self_employed_encoder.classes_:
            return render_template('Loan_approval.html', prediction_text="Invalid input in categorical fields.")

        education_encoded = education_encoder.transform([education])[0]
        self_employed_encoded = self_employed_encoder.transform([self_employed])[0]

        # Combine all inputs into a single array
        features = np.array([[education_encoded, self_employed_encoded, income, loan_term, cibil_score,loan_amount,
                              residential_assets_value, commercial_assets_value]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template('Loan_approval.html', prediction_text=f"Loan Status: {'Approved' if prediction==1 else 'Rejected'}")

    except Exception as e:
        return render_template('Loan_approval.html', prediction_text=f"Error occurred: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
