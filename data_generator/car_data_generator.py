import pandas as pd
import numpy as np


class CarDataGenerator:
    @staticmethod
    def generate_car_data(num_samples=1000):
        np.random.seed(42)
        data = {
            "mileage": np.random.randint(5000, 200000, num_samples),
            "age": np.random.randint(1, 20, num_samples),
            "horsepower": np.random.randint(60, 300, num_samples),
            "price": np.random.randint(5000, 50000, num_samples),
        }
        return pd.DataFrame(data)

    @staticmethod
    def save_to_csv(data, filename="data/car_data.csv"):
        data.to_csv(filename, index=False)
