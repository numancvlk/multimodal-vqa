# LIBRARIES
from PIL import Image
from app.models.blip import BlipModel
from app.models.clip import ClipModel
from app.models.llama import LlamaModel

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

    # 3. ADIM MODELE GEREKLI BILGILERI VERIYORUM
    prompt = f"""
    You are a helpful AI assistant capable of analyzing images.
    I will provide you with a description of an image (Caption) and a User Question.

    CONTEXT:
    - Image Description: "{caption}"
    - System Confidence Score: {confidenceScore:.2f} (Internal metric, do not mention this to the user)

    INSTRUCTIONS:
    1. Answer the user's question DIRECTLY and CLEARLY based on the Image Description.
    2. Assume the Image Description is correct. Do not apologize or say "I am not sure".
    3. Do NOT mention the confidence score or the internal process in your final answer.
    4. If the question asks for a count (how many), use the description to estimate.

    User Question: {question}
    
    Answer:
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