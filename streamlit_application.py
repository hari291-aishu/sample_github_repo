# app.py
import streamlit as st
import pickle
import numpy as np

# Load the trained model (use joblib if you've saved with joblib)
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set the title for the app
st.title('Random Forest Model Prediction App')

# Add a description or instructions
st.write("""
    This app uses a trained Random Forest model to predict the class of the Iris dataset.
    Please input the features of the flower, and the model will predict the species.
""")

# Input fields for the user to enter data
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.5)

# Make a prediction when the user clicks the button
if st.button('Predict'):
    # Create a NumPy array for the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get prediction from the model
    prediction = model.predict(input_data)
    
    # Map the prediction to the corresponding class
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_class = species[prediction[0]]
    
    # Display the result
    st.write(f'The predicted species is: {predicted_class}')
