from pydantic import BaseModel

class Settings(BaseModel):
    model_path: str = "source/rf_kaggle.joblib"
    train_path: str = "source/train.csv"
    test_path: str = "source/test.csv"

settings = Settings()