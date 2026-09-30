from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings

# Ignore scikit-learn version warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get inputs from the form
        Car_Name = request.form.get('Car_Name', 'Car').title()
        
        Year = int(request.form['Year'])
        Year = 2021 - Year  # Convert to age of car
        
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        
        Fuel_Type = request.form['Fuel_Type']
        Fuel_Type_Diesel = 1 if Fuel_Type == 'Diesel' else 0
        Fuel_Type_Petrol = 1 if Fuel_Type == 'Petrol' else 0
        
        Seller_Type = request.form['Seller_Type']
        Seller_Type_Individual = 1 if Seller_Type == 'Individual' else 0
        
        Transmission = request.form['Transmission']
        Transmission_Manual = 1 if Transmission == 'Manual' else 0
        
        # Prepare feature array
        features = np.array([[Year, Present_Price, Kms_Driven, Owner, 
                              Fuel_Type_Diesel, Fuel_Type_Petrol, 
                              Seller_Type_Individual, Transmission_Manual]])
        
        # Make Prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        
        if output < 0:
            return render_template('index.html', prediction_text=f"Sorry, you cannot sell your {Car_Name} (Estimated Value is negative).")
        else:
            return render_template('index.html', prediction_text=f"Estimated Selling Price for your {Car_Name}: {output} Lakhs")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
