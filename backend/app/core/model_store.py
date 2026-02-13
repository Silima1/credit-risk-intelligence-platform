import joblib
import pandas as pd
from .config import settings
from app.services.features import add_date_features, get_feature_columns

class ModelStore:
    def __init__(self):
        self.model = None
        self.train_df = None
        self.test_df = None
        self.feature_cols = None

    def load(self):
        self.model = joblib.load(settings.model_path)
        self.train_df = pd.read_csv(settings.train_path)
        self.test_df = pd.read_csv(settings.test_path)

        if "application_date" in self.train_df.columns:
            self.train_df = add_date_features(self.train_df)
            self.test_df = add_date_features(self.test_df)

        self.feature_cols = get_feature_columns(self.model, self.train_df)

store = ModelStore()