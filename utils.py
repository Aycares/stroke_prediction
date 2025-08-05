from zenml.client import Client
from xgboost import XGBClassifier
import xgboost as xgb
import logging
from logging import getLogger
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()

model_id = os.getenv(key="MODEL_ARTIFACT")
scaler_id = os.getenv(key="SCALER_ARTIFACT")
encoder_id = os.getenv(key="ENCODER_ARTIFACT")

# configure logging
logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

# get the columns in the training_data
def get_columns() -> List[str]:
    artifact = Client().get_artifact_version("pylf_v1_us_gCNfc7tDjk7346hYHm62TjvtW88wVBLzQvq6NbpsLb49")
    data = artifact.load()
    column_names = list(data.columns)
    
    return column_names

# get the scaler, encoder and model objects.
def get_artifacts() -> Tuple[Dict, StandardScaler, XGBClassifier]:
    """This function returns the encoder dictionary,
    model, standard scaler object and model artifact
    associated with the prod training pipeline."""
    
    # encoder
    artifact = Client().get_artifact_version(str(encoder_id))
    label_encoders = artifact.load()
    # scaler
    artifact = Client().get_artifact_version(str(scaler_id))
    scaler = artifact.load()
    # model
    artifact = Client().get_artifact_version(str(model_id))
    model = artifact.load()
    
    return label_encoders, scaler, model

# write a function that makes prediction

def predict_stroke(data: Dict) -> float:
    final_data = {column:[value] for column, value in data.items()}
    logger.info(f"data: {final_data}")
    final_data = pd.DataFrame(final_data)
    label_encoders, scaler, model = get_artifacts()
    for column_name in label_encoders.keys():
        final_data[column_name] = label_encoders[column_name].transform(final_data[column_name])
    # scale the dataset
    columns = list(final_data.columns)
    final_data = scaler.transform(final_data)
    final_data = pd.DataFrame(data=final_data, columns=columns)
    
    # get prediction
    pred = model.predict(final_data)
    
    return pred[0]

if __name__ == "__main__":
    data = {
        "gender": "Male",
        "age": 45,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 105.5,
        "bmi": 28.4,
        "smoking_status": "never smoked"
    }
    print(predict_stroke(data=data))
