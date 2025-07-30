from fastapi.datastructures import UploadFile
from typing import Annotated
from api.use_cases.vision_uc import validate_laptop_name
from fastapi import APIRouter, Request
import pandas as pd
from api.contracts.price_contracts import PredictRequest
from api.use_cases.price_uc import preprocess_request
from fastapi import File, Form

router = APIRouter(prefix="/models", tags=["Models"])

@router.post("/price")  
async def predict_price(
    body: PredictRequest,
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
    description: str = Form(...),
):
    await validate_laptop_name(image, description)
    return {"detail": "La imagen es válida."}

@router.post("/gama")
async def predict_gama(body: PredictRequest, request: Request):
    df_input = await preprocess_request(body)
    gama_model = request.app.state.models["gama_model"]
    pred = gama_model.predict(df_input)[0]

    mapping = {0: "baja", 1: "media", 2: "alta"}
    return {
        "gama_code": int(pred),
        "gama_label": mapping[int(pred)]
    }

    return {"predicted_gama": round(float(pred), 2)}

@router.post("/priceperformance")
async def predict_priceperformance(body: PredictRequest, request: Request):
    df_input = await preprocess_request(body)
    priceperformance_model = request.app.state.models["priceperformance_model"]
    pred = priceperformance_model.predict(df_input)[0]

    mapping = {0: "no", 1: "yes"}

    return {"predicted_priceperformance": round(float(pred), 2), "priceperformance_label": mapping[int(pred)]}