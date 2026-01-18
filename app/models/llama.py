# LIBRARIES
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# HYPERPARAMETERS
MODEL_NAME = os.getenv("LLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")

class LlamaModel:
    def __init__(self):
        self.ollamaURL = OLLAMA_URL
        self.modelName = MODEL_NAME

    def generate_answers(self, prompt:str) -> str:
        input = {
            "model": self.modelName,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(f"{self.ollamaURL}/api/generate", json=input)

            response.raise_for_status() #HATA ALMAK ICIN KOYDUM

            return response.json().get("response","")
        
        except:
            return "MODEL HATASI"