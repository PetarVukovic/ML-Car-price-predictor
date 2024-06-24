import torch


class CarPricePredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, mileage, age, horsepower):
        self.model.eval()
        features = torch.tensor(
            self.scaler.transform([[mileage, age, horsepower]]), dtype=torch.float32
        )
        with torch.no_grad():
            prediction = self.model(features).item()
        return prediction
