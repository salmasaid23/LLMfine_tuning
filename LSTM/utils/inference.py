from typing import List, Dict
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.config import best_bilstm, img_tokenizer, query_tokenizer, label_enc,IMG_MAX_LENGTH,QUERY_MAX_LENGTH
from utils.text_processor import TextProcessor

class ToxicityClassifier:
    def __init__(self):
        self.processor = TextProcessor()
        self.model = best_bilstm
        self.img_tokenizer = img_tokenizer
        self.query_tokenizer = query_tokenizer
        self.label_encoder = label_enc

    def predict(self, image_descriptions: List[str], queries: List[str]) -> List[Dict]:

        cleaned_descriptions = [self.processor.clean_text(text) for text in image_descriptions]
        cleaned_queries = [self.processor.clean_text(text) for text in queries]

        tokenized_descriptions = [word_tokenize(text) for text in cleaned_descriptions]
        tokenized_queries = [word_tokenize(text) for text in cleaned_queries]
        
        img_sequences = self.img_tokenizer.texts_to_sequences(tokenized_descriptions)
        query_sequences = self.query_tokenizer.texts_to_sequences(tokenized_queries)

        img_padded = pad_sequences(img_sequences, maxlen=IMG_MAX_LENGTH, padding='post', truncating='post')
        query_padded = pad_sequences(query_sequences, maxlen=QUERY_MAX_LENGTH, padding='post', truncating='post')

        predictions = self.model.predict([img_padded, query_padded])
        predicted_classes = predictions.argmax(axis=1)

        results = []
        for idx, (img_desc, query, probs, pred_class) in enumerate(
            zip(image_descriptions, queries, predictions, predicted_classes)
        ):
            class_name = self.label_encoder.classes_[pred_class]
            confidence_score = float(probs[pred_class])
            
            result_dict = {
                "image_description": img_desc,
                "search_query": query,
                "primary_toxicity_class": class_name,
                "confidence_score": confidence_score
            }
            results.append(result_dict)
        
        return results