from fastapi import FastAPI, HTTPException, status
from xgboost import XGBClassifier
import xgboost as xgb
from utils import predict_stroke
from schema import RootResponse,ModelRequest,ModelResponse
import logfire
from dotenv import load_dotenv
import os
import uvicorn


# load the environment variables
load_dotenv()
logfire_token = os.getenv(key="LOGFIRE_TOKEN")

# initialize the app
app = FastAPI(title="Endpoint For stroke Prediction",
              version= "v1")

logfire.configure(token=logfire_token)
logfire.instrument_fastapi(app=app)

# create the root endpoint
@app.get(path="/", tags=["Root Endpoints"], response_model=RootResponse)
def root():
    """This endpoint serves the root api!"""
    return RootResponse(message="We are live!")

# create the prediction endpoint
@app.post(path="/predict/", tags=["stroke"], response_model=ModelResponse)
def get_stroke(payload: ModelRequest):
    """This is the endpoint for the model prediction."""
    try:
        data = payload.model_dump()
        stroke = predict_stroke(data)
        logfire.info(f"predicted_stroke: {stroke}, payload: {data}")
        
        return ModelResponse(
            predicted_stroke=stroke
        )
    except Exception as err:
        logfire.error(f"An error occured. Details: {err}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"{err}")


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="localhost", port=8000, reload=True)