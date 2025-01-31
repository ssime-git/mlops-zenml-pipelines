import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple
from typing_extensions import Annotated

# zenml importing
from zenml import step, pipeline
from sklearn.base import ClassifierMixin
from zenml.client import Client

# Get experiment tracker if available, otherwise None
experiment_tracker = Client().active_stack.experiment_tracker

def get_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray):
   """Calculate classification metrics."""
   accuracy = accuracy_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred, average='macro')
   recall = recall_score(y_true, y_pred, average='macro')
   f1 = f1_score(y_true, y_pred, average='macro')
   return accuracy, precision, recall, f1

@step
def load_data(data_path: str) -> pd.DataFrame:
   """Load a dataset."""
   try:
       data = pd.read_csv(data_path)
       return data
   except Exception as e:
       logging.error(f"Error loading data: {e}")
       raise e

@step
def data_preparation(
   data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   """Prepare data for training."""
   try:
       dataframe = data.copy()
       dataframe['age'] = round(dataframe['age'] / 365).astype(int)

       X = data.drop('cardio', axis=1).values
       y = data['cardio'].values
       X_train, X_test, y_train, y_true = train_test_split(
           X, y, test_size=0.2, random_state=42
       )
       return X_train, X_test, y_train, y_true
   except Exception as e:
       logging.error(f"Error in data preparation: {e}")
       raise e

@step
def get_model_config() -> dict:
   """Get model configuration."""
   return {
       "model_name": "model",
       "model_params": {
           'max_depth': 4, 
           'random_state': 42
       }
   }

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def train_rf(
   X_train: np.ndarray, 
   y_train: np.ndarray, 
   config: dict
) -> ClassifierMixin:
   """Train a Random Forest model."""
   try:
       params = config["model_params"]
       model = RandomForestClassifier(**params)
       model.fit(X_train, y_train)

       # mlflow logging
       mlflow.sklearn.log_model(model, config["model_name"])
       for param, value in params.items():
           mlflow.log_param(param, value)

       return model
   except Exception as e:
       logging.error(f"Error in training model: {e}")
       raise e

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def evaluate_model(
   model: ClassifierMixin, 
   X_test: np.ndarray, 
   y_test: np.ndarray
) -> Tuple[
   Annotated[float, "accuracy"],
   Annotated[float, "recall"]
]:
   """Evaluate model and log metrics."""
   try:
       y_pred = model.predict(X_test)

       # metrics
       accuracy, precision, recall, f1 = get_clf_metrics(y_test, y_pred)
       
       mlflow.log_metric("accuracy", accuracy)
       mlflow.log_metric("precision", precision)
       mlflow.log_metric("recall", recall)
       mlflow.log_metric("f1", f1)

       return accuracy, recall
   except Exception as e:
       logging.error(f"Error in evaluating model: {e}")
       raise e

@pipeline
def training_rf_pipeline(data_path: str):
   """Pipeline to train and evaluate a Random Forest model."""
   data = load_data(data_path=data_path)
   X_train, X_test, y_train, y_test = data_preparation(data=data)
   config = get_model_config()
   model = train_rf(X_train=X_train, y_train=y_train, config=config)
   accuracy, recall = evaluate_model(model=model, X_test=X_test, y_test=y_test)
   print(f"Accuracy: {accuracy}, Recall: {recall}")

def main():
   """Main function to run the pipeline."""
   # Make sure this path is correct relative to where the script is run
   training_rf_pipeline(data_path="data/cardio_train_sampled.csv")

if __name__ == '__main__':
   main()