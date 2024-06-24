import pandas as pd
import numpy as np


# Generisanje sintetičkih podataka o cenama automobila
def generate_car_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "mileage": np.random.randint(5000, 200000, num_samples),
        "age": np.random.randint(1, 20, num_samples),
        "horsepower": np.random.randint(60, 300, num_samples),
        "price": np.random.randint(5000, 50000, num_samples),
    }
    return pd.DataFrame(data)


# Spremanje dataset-a u CSV fajl
car_data = generate_car_data()
car_data.to_csv("car_data.csv", index=False)
print(car_data.head())

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Definicija PyTorch modela
class CarPriceModel(nn.Module):
    def __init__(self):
        super(CarPriceModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Funkcija za treniranje modela
def train_model(data):
    X = data[["mileage", "age", "horsepower"]].values
    y = data["price"].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = CarPriceModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

    return model, scaler


# Treniranje modela sa generisanim podacima
model, scaler = train_model(car_data)
import streamlit as st


# Funkcija za predikciju nove cene
def predict_price(model, scaler, mileage, age, horsepower):
    model.eval()
    features = torch.tensor(
        scaler.transform([[mileage, age, horsepower]]), dtype=torch.float32
    )
    with torch.no_grad():
        prediction = model(features).item()
    return prediction


# Streamlit interfejs
def main():
    st.title("Predikcija cena automobila koristeći PyTorch")

    st.write("### Unesite podatke o automobilu za predikciju cene")

    mileage = st.number_input("Kilometraža", min_value=0, max_value=300000, value=50000)
    age = st.number_input(
        "Starost automobila (u godinama)", min_value=0, max_value=30, value=5
    )
    horsepower = st.number_input(
        "Snaga motora (u konjskim snagama)", min_value=50, max_value=500, value=150
    )

    if st.button("Predvidi cenu"):
        prediction = predict_price(model, scaler, mileage, age, horsepower)
        st.write(f"Procijenjena cena automobila je: ${prediction:.2f}")


if __name__ == "__main__":
    main()
