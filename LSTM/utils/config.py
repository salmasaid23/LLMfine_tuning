import os
import joblib
from dotenv import load_dotenv
from tensorflow.keras.models import load_model


# Load environment variables from .env file
load_dotenv(override=True)

# Variables
APP_NAME = os.getenv('APP_NAME')
VERSION = os.getenv('VERSION')
DESCRIPTION=os.getenv('DESCRIPTION')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_FOLDER_PATH = os.path.join(BASE_DIR, "assets")

# Load models
img_tokenizer = joblib.load(os.path.join(ASSETS_FOLDER_PATH,'img_tokenizer.joblib'))
query_tokenizer = joblib.load(os.path.join(ASSETS_FOLDER_PATH,'query_tokenizer.joblib'))
label_enc = joblib.load(os.path.join(ASSETS_FOLDER_PATH,'label_encoder.joblib'))
best_bilstm = load_model(os.path.join(ASSETS_FOLDER_PATH,'best_bilstm_model.keras'))

# constants
IMG_MAX_LENGTH = 7
QUERY_MAX_LENGTH = 15
IMG_VOCAB_SIZE = 59
QUERY_VOCAB_SIZE = 3275
IMG_EMBEDDING_DIM = 64
QUERY_EMBEDDING_DIM = 128
LSTM_UNITS = 64

# Toxicity classes mapping 
TOXICITY_MAPPING = {
    0: "toxic",
    1: "severe_toxic",
    2: "obscene",
    3: "threat",
    4: "insult",
    5: "identity_hate",
    6: "sexual_explicit",
    7: "spam",
    8: "non_toxic"
}
