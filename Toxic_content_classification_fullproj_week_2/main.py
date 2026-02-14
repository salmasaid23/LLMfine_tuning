from utils.distilbert_classifier import predict
from utils.sqlite_db import save_to_database
from utils.imagecaption import img_captioning_generation
from utils.sqlite_db import initialize_database
# from utils.test_img_app_ui import run_app
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Initialize DB on startup
initialize_database()
#classify and save input data to data base
def process_text_input(text: str):

    label, confidence = predict(text)

    save_to_database(
        input_type="text",
        user_input=text,
        prediction=label
    )

    return label, confidence

def process_image_input(image_file):

    caption = img_captioning_generation(image_file)
    label, confidence = predict(caption)

    save_to_database(
        input_type="image",
        user_input=caption,
        prediction=label
    )
    return caption, label, confidence

# if __name__ == "__main__":
#     run_app()