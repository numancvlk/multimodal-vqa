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


    ai_attitude = """
    1. Answer the user's question DIRECTLY and CLEARLY based on the Image Description.
    2. Assume the Image Description is correct. Do not apologize.
    """

    if confidenceScore < 0.25:
        ai_attitude = """
        1. The image description might be INACCURATE due to low confidence.
        2. Answer cautiously. You can say 'It appears to be...' or 'It might be...'.
        3. If the user asks for specific details, warn them that the image is unclear.
        """

    # 3. ADIM MODELE GEREKLI BILGILERI VERIYORUM
    prompt = f"""
    You are a helpful AI assistant analyzing an image.
    
    CONTEXT:
    - Image Description: "{caption}"
    - Confidence Score: {confidenceScore:.2f}

    INSTRUCTIONS:
    {ai_attitude}
    
    3. Do NOT mention the confidence score number in your final answer.
    4. If the question asks for a count, estimate based on description.

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