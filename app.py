from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/model.pkl', 'rb'))

# HTML Template
template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Loan Default Predictor</title>
    <style>
        body { font-family: Arial; background-color: #f0f0f0; padding: 40px; }
        form { background: #fff; padding: 20px; border-radius: 10px; max-width: 400px; margin: auto; }
        input, select { width: 100%; padding: 10px; margin: 8px 0; border-radius: 4px; border: 1px solid #ccc; }
        button { background-color: #28a745; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #218838; }
        h2, .result { text-align: center; }
    </style>
</head>
<body>
    <h2>Loan Default Prediction</h2>
    <form method="POST">
        <input type="number" name="Age" placeholder="Age" required>
        <input type="number" name="Income" placeholder="Income" required>
        <input type="number" name="LoanAmount" placeholder="Loan Amount" required>
        <input type="number" name="CreditScore" placeholder="Credit Score" required>
        <select name="HasMortgage" required>
            <option value="Yes">Has Mortgage</option>
            <option value="No">No Mortgage</option>
        </select>
        <select name="HasDependents" required>
            <option value="Yes">Has Dependents</option>
            <option value="No">No Dependents</option>
        </select>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
        <div class="result">
            <p><strong>Prediction:</strong> {{ 'Will Default' if prediction==1 else 'No Default' }}</p>
        </div>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get and process input values
        age = int(request.form['Age'])
        income = int(request.form['Income'])
        loan_amount = int(request.form['LoanAmount'])
        credit_score = int(request.form['CreditScore'])
        has_mortgage = 1 if request.form['HasMortgage'] == 'Yes' else 0
        has_dependents = 1 if request.form['HasDependents'] == 'Yes' else 0

        features = np.array([[age, income, loan_amount, credit_score, has_mortgage, has_dependents]])
        prediction = model.predict(features)[0]

    return render_template_string(template, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
 