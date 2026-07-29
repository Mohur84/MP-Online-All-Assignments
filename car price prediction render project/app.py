# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved Random Forest model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form input fields using their exact HTML name attributes
            present_price = float(request.form['present_price'])
            kms_driven = int(request.form['kms_driven'])
            fuel_type = int(request.form['fuel_type'])
            seller_type = int(request.form['seller_type'])
            transmission = int(request.form['transmission'])
            owner = int(request.form['owner'])
            car_age = int(request.form['car_age'])

            # Restructure inputs into a 2D array for the scikit-learn model
            features = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age]])
            
            # Predict price using the model pipeline
            prediction = model.predict(features)[0]
            output = round(prediction, 2)

            return render_template('index.html', prediction_text=f'Estimated Car Price: ₹ {output} Lakhs')
        except Exception as e:
            return render_template('index.html', prediction_text="Error: Please verify your input parameters.")

if __name__ == '__main__':
    app.run(debug=True)