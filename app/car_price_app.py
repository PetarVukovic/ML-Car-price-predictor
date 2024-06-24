import streamlit as st
from models.car_price_predictor import CarPricePredictor


class CarPriceApp:
    def __init__(self, predictor):
        self.predictor = predictor

    def run(self):
        st.title("Car Price Prediction using PyTorch")

        st.write("### Enter car details for price prediction")

        mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000)
        age = st.number_input("Car Age (in years)", min_value=0, max_value=30, value=5)
        horsepower = st.number_input(
            "Engine Power (in horsepower)", min_value=50, max_value=500, value=150
        )

        if st.button("Predict Price"):
            prediction = self.predictor.predict(mileage, age, horsepower)
            st.write(f"Estimated car price: ${prediction:.2f}")
