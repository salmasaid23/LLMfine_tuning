# install pillow for image captioning 10.3.0
# install transformers 4.41.2
# pil used for opening and processing image files 
from transformers import  BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


#load the processor and the model 
processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#Generate image caption function
def img_captioning_generation(img, text="a photo of"):
    image = Image.open(img).convert('RGB')
    
    inputs = processor(image, text, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption