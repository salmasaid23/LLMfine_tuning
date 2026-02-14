import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import LLAMA_GUARD_TOKEN, LABELS


@st.cache_resource # Load this ONCE and reuse it across reruns
def load_llama_guard():
    model_name = "meta-llama/Llama-Guard-3-1B"
     
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=LLAMA_GUARD_TOKEN, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",token=LLAMA_GUARD_TOKEN, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    return tokenizer, model

tokenizer, model = load_llama_guard()

def build_prompt(content: str) -> str:
    labels_text = "\n".join([f"- {label}" for label in LABELS])

    prompt = f"""
You are a content safety classifier.

Classify the following content into EXACTLY ONE of the categories below.
You MUST choose one label from the list and output ONLY the label text.

Categories:
{labels_text}

Content:
{content}

Answer:
"""
    return prompt.strip()


def _extract_label(generated_text: str) -> str:
    """
    Ensure output is EXACTLY one label from LABELS.
    We'll search for any label occurrence and return the first match.
    If none found, return 'Unknown S-Type'.
    """
    # Normalize a bit
    text = (generated_text or "").strip()

    # Look for exact label lines
    for label in LABELS:
        if label in text:
            return label

    # Fallback
    return "Unknown S-Type"


def classify_text(content: str) -> str:
    prompt = build_prompt(content)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,max_length=512).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    decoded_output = tokenizer.decode(output[0],skip_special_tokens=True)
        # Only keep the part after "Answer:" if present
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:", 1)[-1].strip()

    return decoded_output