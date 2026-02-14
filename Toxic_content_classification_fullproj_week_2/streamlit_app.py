import streamlit as st
from main import process_text_input, process_image_input
from utils.sqlite_db import view_all_records_dataframe
from utils.config import APP_NAME, VERSION, DESCRIPTION

st.set_page_config(
    page_title=APP_NAME,
    layout="centered"
)

st.title(APP_NAME)
st.caption(f"Version {VERSION}")
st.write(DESCRIPTION)

st.divider()

# TEXT INPUT SECTION
st.subheader("Text Classification")

text_input = st.text_area("Enter text")

if st.button("Classify Text"):
    if text_input.strip():
        label, confidence = process_text_input(text_input)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text.")

st.divider()

# IMAGE INPUT SECTION
st.subheader("Image Classification")

uploaded_image = st.file_uploader("Upload an image",type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        caption, label, confidence = process_image_input(uploaded_image)

        st.write(f"**Generated Caption:** {caption}")
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2f}")

st.divider()

# DATABASE VIEW
st.subheader("Stored Records")

if st.button("View Database"):
    df = view_all_records_dataframe()
    st.dataframe(df, use_container_width=True)
