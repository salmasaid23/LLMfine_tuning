## `LSTM Toxic Content Classification `

A deep learning project using **LSTM** model to classify toxic content in text queries and image descriptions.
This model helps identify toxic content across multiple categories.

## Project Structure

```
project/
├── main.py
├── .env
├── requirements.txt
├── utils/
│   ├── config.py
│   ├── inference.py
│   ├── schemas.py
│   └── text_processor.py
└── assets/
    ├── best_bilstm_model.keras
    ├── img_tokenizer.joblib
    ├── query_tokenizer.joblib
    └── label_encoder.joblib
```

## Install dependencies:

```bash
$ pip install -r requirements.txt
```

## Run the FastAPI server

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### The API will be available at `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check

```bash
GET /health
Header: X-API-Key: your-secret-key
```

### 2. Single Prediction

```bash
POST /predict
Header: X-API-Key: your-secret-key

Body:
{
    "image_description": "violent scene",
    "query": "how to hurt people"
}

Response:
{
    "predicted_category": "threat",
    "confidence": 0.92
}
```

### 3. Batch Prediction

```bash
POST /predict-batch
Header: X-API-Key: your-secret-key

Body:
{
    "items": [
        {
            "image_description": "violent", 
            "query": "hurt"
            },
        {
            "image_description": "sunset",
             "query": "photography"
              }
    ]
}

Response:
{
    "count": 2,
    "predictions": [
        {
            "predicted_category": "threat",
             "confidence": 0.92
             },
        {
            "predicted_category": "non_toxic",
             "confidence": 0.93
             }
    ]
}
```

---

## Toxicity Classes

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate
- sexual_explicit
- spam
- non_toxic

---
