from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    Status: Literal["New", "Refurbished"]
    Brand: str
    Model: str
    CPU: str
    RAM: int
    Storage: int
    Storage_type: Literal["SSD", "eMMC"]
    GPU: str
    Screen: float
    Touch: Literal["Yes", "No"]