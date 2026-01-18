# LIBRARIES
import torch
import os
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

load_dotenv()

# HYPERPARAMETERS
MODEL_NAME = os.getenv("CLIP_MODEL") 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ClipModel:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

    def similarity_score(self, image: Image.Image, text:str) -> float:
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors = "pt",
            padding = True
        ).to(DEVICE)

        outputs = self.model(**inputs)

        imageEmbeds = outputs.image_embeds
        textEmbeds = outputs.text_embeds
        
        imageEmbeds = imageEmbeds / imageEmbeds.norm(p=2, dim=-1, keepdim=True)
        textEmbeds = textEmbeds / textEmbeds.norm(p=2, dim=-1, keepdim=True)

        similarityScore = (imageEmbeds @ textEmbeds.T).item()

        return similarityScore
