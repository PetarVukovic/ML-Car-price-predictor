from data_generator.car_data_generator import CarDataGenerator
from models.car_price_model import CarPriceModel
from models.car_price_trainer import CarPriceTrainer
from models.car_price_predictor import CarPricePredictor
from app.car_price_app import CarPriceApp

if __name__ == "__main__":
    car_data = CarDataGenerator.generate_car_data()
    CarDataGenerator.save_to_csv(car_data)

    model = CarPriceModel()
    trainer = CarPriceTrainer(model, car_data)
    trained_model, scaler = trainer.train()

    predictor = CarPricePredictor(trained_model, scaler)
    app = CarPriceApp(predictor)
    app.run()
