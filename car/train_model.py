import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv('car_data.csv')

# Preprocessing to match the Jupyter Notebook
current_year = 2021
df['Year'] = current_year - df['Year']
df.drop(columns=['Car_Name'], inplace=True)

newdf = pd.get_dummies(df, drop_first=True)

# Features and target variable
expected_cols = ['Year', 'Present_Price', 'Kms_Driven', 'Owner', 
                 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 
                 'Transmission_Manual']
X = newdf[expected_cols]
y = newdf['Selling_Price']

# Train the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully as model.pkl")
