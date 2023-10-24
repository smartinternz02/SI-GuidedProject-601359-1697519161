from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("car_purchase.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Extracting the input features from the form
    Age = int(request.form['Age'])
    AnnualSalary = int(request.form['AnnualSalary'])
    Gender_Male = int(request.form['Gender_Male'])
    Gender_Female = int(request.form['Gender_Female'])

    # Create a numpy array for prediction
    input_features = np.array([[Age, AnnualSalary, Gender_Male, Gender_Female]])

    # Make a prediction using the loaded model
    prediction = model.predict(input_features)

    # Assuming prediction is a 1D array, you can access the first element directly
    probability_of_purchase = prediction[0]

    if probability_of_purchase > 0.5:
        return render_template('car_purchase.html', pred=f'Probability of this person purchasing a car is {probability_of_purchase:.2f}', xyz='Convert this potential customer')
    else:
        return render_template('car_purchase.html', pred=f'Probability of this person purchasing a car is {probability_of_purchase:.2f}', xyz='Try to spend fewer marketing resources here')

if __name__ == '__main__':
    app.run(debug=True)

