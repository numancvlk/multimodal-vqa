# LIBRARIES
import os
from dotenv import load_dotenv
from PIL import Image
from app.models.blip import BlipModel
from app.models.clip import ClipModel
from app.models.llama import LlamaModel

load_dotenv()

# HYPERPARAMETERS
THRESHOLD = os.getenv("CLIP_THRESHOLD")

blip_model = BlipModel()
clip_model = ClipModel()
llama_model = LlamaModel()

def process_request(image: Image.Image, question: str):
    #1. ADIM BLIP ILE RESME NAKTIK
    caption = blip_model.generate_captions(
        image=image
    )

    #2. ADIM CLIP ILE BLIPIN GORDUGU SEY AYNIMI DIYE KONTROL EDIYORUZ
    confidenceScore = clip_model.similarity_score(
        image= image,
        text= caption
    )

    threshold = float(THRESHOLD)

    warnMessage = ""

    if confidenceScore < threshold:
        warnMessage = "(WARNING: The visual could not be fully understood; approach this information with caution.)"

    # 3. ADIM MODELE GEREKLI BILGILERI VERIYORUM
    prompt = f"""
    You are an intelligent assistant with visual perception capabilities.
    Below, there is a description (caption) of an image and the user's question.

    Image Description: {caption}
    Accuracy Confidence: {confidenceScore:.2f} {warnMessage}

    User Question: {question}

    Please provide a logical and helpful answer based on the image description.
    Give the answer in English.
    """

    # 4. ADIM CEVAP URETIYORUZ BU KISIMDA
    finalAnswer = llama_model.generate_answers(
        prompt= prompt
    )

    # BURDA ISE TUM MODELLERDEN ELDE ETTIGIMIZ HER SEYI ALIP FRONTEND DE GOSTERMEK ICIN RETURN ETTIM
    return {
        "caption": caption,
        "confidence_score": confidenceScore,
        "answer": finalAnswer
    }