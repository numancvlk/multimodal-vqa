# LIBRARIES
import io
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

from app.services.allservice import process_request
from app.schemas import ResponseModel

app = FastAPI(
    title="Multimodal Image QA",
    description= "CLIP + BLIP + LLAMA LOCAL PIPELINE"
)

@app.post("/predict", response_model= ResponseModel)
async def predict(
    image: UploadFile = File(),
    question: str = Form()
):
    
    image = await image.read() #GELEN RESIM SAYIALRDAN OLUSTUGU ICIN ONU BURDA OKUYACAGIM

    pilImage = Image.open(io.BytesIO(image)).convert("RGB") #BURDA ISE GELEN SAYIALRI PIL FORMATINA CEVIRIYORUM

    result = process_request(
        image= pilImage,
        question= question
    )

    return result