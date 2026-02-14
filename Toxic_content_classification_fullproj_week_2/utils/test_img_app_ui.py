import streamlit as st
from .imagecaption import img_captioning_generation

def run_app():

    st.set_page_config(page_title="Image Captioning")

    # Title
    st.title("Image Captioning App")
    st.write("Upload an image and generate a caption using a pre-trained BLIP model.")
    st.divider()

    # Upload image
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file",type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_column_width=True
        )

    st.divider()

    # Prompt
    st.header("Caption Prompt")
    prompt = st.text_input("Optional text prompt",value="a photo of")

    st.divider()

    # Generate caption
    st.header("Generate Caption")

    if uploaded_file:
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = img_captioning_generation(uploaded_file, prompt)

            st.success("Caption generated!")
            st.write("**Caption:**")
            st.write(caption)
    else:
        st.info("Please upload an image to enable caption generation.")

    st.divider()
    st.caption("Model: Salesforce BLIP | Framework: Streamlit")