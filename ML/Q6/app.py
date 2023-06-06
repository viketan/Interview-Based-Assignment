from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
y_encoder = joblib.load('y_encoder.pkl')
X_encoder = joblib.load('X_encoder.pkl')

def preprocessing(input):
    input_encoded = X_encoder.transform(input[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']])
    input_scaler = scaler.transform(input[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',                                   
       'Loan_Amount_Term', 'Credit_History']])
    input = np.concatenate((input_encoded,input_scaler),axis=1)
    return input

def decode(input):
    return y_encoder.inverse_transform(input)



app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    name = request.form['name']
    gender = request.form['gender']
    married = request.form['married']
    dependents = request.form['dependents']
    education = request.form['education']
    self_employed = request.form['selfEmployed']
    applicant_income = float(request.form['applicantIncome'])
    coapplicant_income = float(request.form['coapplicantIncome'])
    loan_amount = float(request.form['loanAmount'])
    loan_amount_term = float(request.form['loanAmountTerm'])
    credit_history = float(request.form['creditHistory'])
    property_area = request.form['propertyArea']
    
    # Perform loan eligibility calculations or use your machine learning model here
    input = [gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area]
    input = pd.DataFrame([input], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    input = preprocessing(input)
    output = model.predict(input)
    eligibility_status = decode(output)[0]
    if eligibility_status == 'Y':
        eligibility_status = 'eligible'
    else:
         eligibility_status = 'Not eligible'
    
    return render_template('result.html', name=name, eligibility_status=eligibility_status)

if __name__ == '__main__':
    app.run(debug=True)
