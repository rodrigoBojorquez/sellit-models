from fastapi.datastructures import UploadFile
from typing import Annotated
from api.use_cases.price_uc import validate_laptop_name
from fastapi import APIRouter, Request
import pandas as pd
from api.contracts.price_contracts import PredictPriceRequest
from api.use_cases.price_uc import preprocess_request
from fastapi import File, Form

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/price")  
async def predict_price(
    body: PredictPriceRequest,
    request: Request,
):
    df_input = await preprocess_request(body)
    price_model = request.app.state.models["price_model"]
    pred = price_model.predict(df_input)[0]

    return {"predicted_price_mxn": round(float(pred), 2)}

@router.post("/validate-laptop")
async def validate_laptop(
    request: Request,
    image: UploadFile = File(...),
    brand: str = Form(...),
    model: str = Form(...),
):
    await validate_laptop_name(image, brand, model)
    return {"detail": "La imagen es válida."}