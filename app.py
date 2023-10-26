import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset here
x = pd.read_excel("output2019_2020_2021.xlsx")
df = pd.DataFrame(x)

# Split the data into features and target
X = df[[2019, 2020]]
y = df[2021]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit app starts here
st.title('Linear Regression Model Deployment')

# Add an input form for user input
st.header('Input Values')
feature1 = st.number_input('2019', min_value = 0.0, step = 0.01)
feature2 = st.number_input('2020', min_value = 0.0, step = 0.01)

# Make predictions with the trained model
if st.button('Predict'):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({'2019': [feature1], '2020': [feature2]})

    # Use the model to make predictions
    prediction = model.predict(input_data)

    st.subheader('Prediction:')
    st.write(f'Predicted Target: {prediction[0]:.2f}')

# Add an evaluation section (optional)
st.header('Model Evaluation')
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
st.write(f'Mean Squared Error (MSE): {mse:.2f}')
st.write(f'R-squared (R2) Score: {r2:.2f}')
