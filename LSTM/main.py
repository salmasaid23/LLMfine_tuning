from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from utils.config import APP_NAME, VERSION, DESCRIPTION, API_SECRET_KEY
from utils.inference import ToxicityClassifier
from utils.schemas import UserDataRequest, PredictionResponse, BatchPredictionRequest


app = FastAPI(
    title=APP_NAME,
    description=DESCRIPTION,
    version=VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = ToxicityClassifier()

api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key


@app.get("/health", tags=['Healthy'], description="Endpoint to check if the API is up and running")
async def health_check(api_key: str=Depends(verify_api_key)):
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "description": DESCRIPTION,
        "status": "up & running"
    }

@app.post("/predict", tags=['Classification'], response_model=PredictionResponse, description="Predict toxicity for a single input")
async def predict_single(request: UserDataRequest, api_key: str=Depends(verify_api_key)):
    try:
        predictions = classifier.predict(
            image_descriptions=[request.image_description],
            queries=[request.query]
        )
        
        pred = predictions[0]
        return PredictionResponse(
            predicted_category=pred["primary_toxicity_class"],
            confidence=pred["confidence_score"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", tags=['Classification'], description="Predict toxicity for multiple inputs")
async def predict_batch(request: BatchPredictionRequest, api_key: str=Depends(verify_api_key)):

    try:
        image_descriptions = [item.image_description for item in request.items]
        queries = [item.query for item in request.items]
        
        predictions = classifier.predict(
            image_descriptions=image_descriptions,
            queries=queries
        )
        
        results = []
        for pred in predictions:
            result = PredictionResponse(
                predicted_category=pred["primary_toxicity_class"],
                confidence=pred["confidence_score"]
            )
            results.append(result)
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

