import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_price_model.pkl')

st.title('House Price Prediction System')

st.write('Enter the details of the house:')

bedrooms = st.number_input('Number of bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of bathrooms', min_value=1, max_value=5, value=2)
kitchens = st.number_input('Number of kitchens', min_value=1, max_value=3, value=1)
lawn = st.selectbox('Presence of a lawn', ['Yes', 'No'])
area_sqft = st.number_input('Total area in square feet', min_value=100, max_value=10000, value=1500)
location_type = st.selectbox('Location type', ['City', 'Out of City'])

if st.button('Predict Price'):
    input_data = {
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'kitchens': [kitchens],
        'lawn': [lawn],
        'area_sqft': [area_sqft],
        'location_type': [location_type]
    }
    import pandas as pd
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    st.success(f'Estimated House Price: ₹{prediction:,.0f}') 