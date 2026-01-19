#LIBRARIES
from pydantic import BaseModel


class ResponseModel(BaseModel):
    caption: str
    confidence_score: float
    answer: str