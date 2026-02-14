import os
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv(override=True)

# Variables
APP_NAME = os.getenv('APP_NAME')
VERSION = os.getenv('VERSION')
DESCRIPTION=os.getenv('DESCRIPTION')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')
LLAMA_GUARD_TOKEN = os.getenv("LLAMA_GUARD_TOKEN")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "assets_of_distilbert")

# for llama_guard
LABELS = [
    "Safe",
    "Violent Crimes",
    "Elections",
    "Sex-Related Crimes",
    #"unsafe", if it is not safe it will be one ao the athor categories
    "Non-Violent Crimes",
    "Child Sexual Exploitation",
    "Unknown S-Type",
    "Suicide & Self-Harm"
]