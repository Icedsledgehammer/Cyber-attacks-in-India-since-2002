import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pk

with open("model.pkl", "rb") as file:
    model = pk.load(file)

# Streamlit app starts here
st.title('Linear Regression Model Deployment')

# Add an input form for user input
st.header('Input Values')
feature1 = st.number_input('2019', min_value = 0, step = 1)
feature2 = st.number_input('2020', min_value = 0, step = 1)

# Make predictions with the trained model
if st.button('Predict'):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({'2019': [feature1], '2020': [feature2]})

    # Use the model to make predictions
    prediction = model.predict(input_data)

    st.subheader('Prediction:')
    st.write(f'Predicted Target: {prediction[0]:.2f}')

