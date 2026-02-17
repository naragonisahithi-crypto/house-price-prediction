import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Predictor")

# User inputs
area = st.number_input("Area (sqft)", 500, 3000, 1000)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])

# Convert location to number
loc_map = {'Urban': 2, 'Semi-Urban': 1, 'Rural': 0}

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, loc_map[location]]],
                              columns=['area','bedrooms','bathrooms','location'])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹ {int(prediction):,}")
