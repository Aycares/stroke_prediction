from pydantic import BaseModel,Field
from typing import Literal


# create a root response object
class RootResponse(BaseModel):
    """creates the payload for the root endpoint."""
    message: str = Field(..., description="root message",
                         examples=["we are live!"])
    

# create a model response object
class ModelResponse(BaseModel):
    """
    creates a response object for the model prediction.
    """
    predicted_charges : float = Field(..., 
                                      description= "The model's predicted stroke",
                                      gt= 0, examples=[55.3, 44.2])
    

    # gender  age  hypertension  heart_disease  ever_married worktype residence avg_glucose_level bmi smoking
# create the model request object
class ModelRequest(BaseModel):
    """
    creates the request object for the model prediction
    """
    age: int = Field(..., description="Age of the client",
                     gt=5, lt=100, examples=[35])
    gender: Literal['male','female'] = Field(..., description="Gender of client",
                                          examples=["male","female"])
    bmi: float = Field(..., description="bmi of the client",
                     gt=5, lt=100, examples=[35.4])  
    
    hypertension: int = Field(..., description="indicates if the client has hypertension (0 = No, 1 = Yes)",
                     ge=0, le=1, examples=[0,1])
    heart_disease: Literal['yes','no'] = Field(..., description="indicates if the client has heart disease (0 = No, 1 = Yes)",
                     ge=0, le=1, examples=[0,1])
    ever_married: Literal['yes', 'no'] = Field(..., description="indicates if the client has ever been married (Yes or No)",
                                          examples=['Yes','No'])
    work_type: Literal['children','govt_job','never worked', 
                    'private', 'self employed'] = Field(..., description="type of work the client is engaged in",
                                          examples=['private','self employed'])
    work_type: Literal['urban','rural',] = Field(..., description="type of resisence the client lives",
                                          examples=['urban','rural'])
    avg_glucose_level: float = Field(..., description="avg glucose level in the blood (mg/dl)",
                     gt=0, lt=500, examples=[95.6])
    smoking_status: Literal['formerly smoked','never smoked',
                            'smokes','unknown'] = Field(..., description="smoking habit of the client",
                                          examples=['never smoked','formerly smoked'])