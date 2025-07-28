from typing import Literal
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
import pandas as pd

router = APIRouter(prefix="/models", tags=["Models"])

class PredictPriceRequest(BaseModel):
    Status: Literal["New", "Refurbished"]
    Brand: str
    Model: str
    CPU: str
    RAM: int
    Storage: int
    Storage_type: Literal["eMMC", "SSD"]
    GPU: str
    Screen: float
    Touch: Literal["Yes", "No"]




@router.post("/price")  
async def predict_price(
    body: PredictPriceRequest,
    request: Request,
):
    data = body.model_dump()
    data["Storage type"] = data.pop("Storage_type")
    df_input = pd.DataFrame([data])
    price_model = request.app.state.models["price_model"]
    pred = price_model.predict(df_input)[0]

    return {"predicted_price_mxn": round(float(pred), 2)}