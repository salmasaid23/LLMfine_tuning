from pydantic import BaseModel, Field
from typing import Literal,List

class UserDataRequest(BaseModel):
    query: str = Field(..., description="The text query to classify", min_length=1)
    image_description: str = Field(..., description="Description of the image", min_length=1)
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How can I create a safe environment for children?",
                "image_description": "A child playing in a sunny meadow."
            }
        }


class PredictionResponse(BaseModel):

    predicted_category: str = Field(..., description="The predicted category")
    confidence: float = Field(..., description="Confidence score of the prediction")
    class Config:
     json_schema_extra = {
        "example": {
          "predicted_category": "Safe",
          "confidence": 0.95
            }
        }

class BatchPredictionRequest(BaseModel):

    items: List[UserDataRequest] = Field(..., description="List of items to classify")
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "query": "How can I create a safe environment for children?",
                        "image_description": "A child playing in a sunny meadow."
                    },
                    {
                        "query": "What are the best methods for home security?",
                        "image_description": "A family enjoying a picnic in the park."
                    }
                ]
            }
        }
        