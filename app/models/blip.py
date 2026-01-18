# LIBRARIES
import torch
import os
from dotenv import load_dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

load_dotenv()

# HYPERPARAMETERS
MODEL_NAME = os.getenv("BLIP_MODEL")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BlipModel:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    
    def generate_captions(self, image: Image.Image) -> str:
        inputs = self.processor( #RESMI ISLEYIP TENSORE CEVIRIYORUZ
            images= image,
            return_tensors = "pt"
        ).to(DEVICE)

        outputs = self.model.generate( #MODELDEN CIKTI ALIYORUZ AMA MAX 50 KELIME OLSUN TOKEN
            **inputs,
            max_new_tokens = 50
        )

        caption = self.processor.decode( #MODEL BIZE SAYISAL CIKTI DONDU BIZ ONU TEKRAR YAZIYA CEVIRIYORUZ
            outputs[0],
            skip_special_tokens = True
        )      

        return caption