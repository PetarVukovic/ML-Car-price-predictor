import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .car_price_model import CarPriceModel
import torch.nn as nn


class CarPriceTrainer:
    def __init__(self, model, data, epochs=100, lr=0.01):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data(data)

    def prepare_data(self, data):
        X = data[["mileage", "age", "horsepower"]].values
        y = data["price"].values.reshape(-1, 1)
        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = self.criterion(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test)
            test_loss = self.criterion(test_outputs, self.y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

        return self.model, self.scaler
