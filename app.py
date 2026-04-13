import streamlit as st 
import numpy as np
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()


#load model and pipeline 
pipeline = joblib.load("pipeline.pkl")

#tittle
st.title("House price prediction App")

st.write("Enter house details below :")

#inputs
# Location
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-110.0, value=-118.0)
latitude = st.number_input("Latitude", min_value=30.0, max_value=45.0, value=34.0)

# Property details
housing_median_age = st.number_input("Property Age", min_value=1, max_value=100, value=25)

total_rooms = st.number_input("Total Rooms", min_value=1, max_value=10000, value=2000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=5000, value=400)

population = st.number_input("Population", min_value=1, max_value=50000, value=1000)
households = st.number_input("Households", min_value=1, max_value=10000, value=300)

median_income = st.number_input("Median Income", min_value=0.0, max_value=20.0, value=5.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

#predict button
if st.button("Predict_price"):

    st.write("✅ Button clicked")

    #prepare input
    input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])
    
    st.write("Input Data:", input_data)


    try:
        processed_data = pipeline.transform(input_data)
        st.write("Processed Data shape:", processed_data.shape)

        prediction = model.predict(processed_data)
        st.write("Raw Prediction:", prediction)
        
        price_usd = prediction[0]
        price_inr = price_usd * 83
        price_crore = price_inr / 10000000

        st.success(f"💰 Estimated Price: ₹ {price_crore:.2f} Crore")

    except Exception as e:
        st.error(f"Error: {e}")

  
