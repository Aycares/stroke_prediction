# import the needed libraries

from zenml import step
import pandas as pd
import numpy as np
from zenml.logger import get_logger
from typing_extensions import Annotated
from typing import Optional,Tuple,Dict
import joblib
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

# configure our logging
logger = get_logger(__name__)

@step
def train_model(X_train: pd.DataFrame, 
                y_train:pd.Series,
                X_test:pd.DataFrame,
                y_test: pd.Series) -> Annotated[Optional[XGBClassifier],
                                                "Model Object"]:
    model = None
    try:
        model = XGBClassifier(random_state=23)
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # compute the scores
        train = f1_score(y_train, train_preds)
        test = f1_score(y_test, test_preds)
        
        logger.info(f"""
                    Completed training the base model with metrics:
                    train rmse: {train}
                    test rmse: {test}
                    """)
    except Exception as err:
        logger.error(f"An error occured. Detail: {err}")
    
    return model