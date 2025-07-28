from fastapi.datastructures import UploadFile
from os import system
from api.common.errors import InvalidLaptopError
from api.common.constants import KNOWN_BY_PRICE_MODEL
from api.contracts.price_contracts import PredictPriceRequest
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from icecream import ic
from api.utils.require_env import require_env
import json
from datetime import datetime
import re
from json import JSONDecodeError
import base64
'''
Preprocesa la request con un prompt a OpenAI para obtener datos mejor adaptados al modelo de predicción.
Devuelve un DataFrame listo para ser usado en la predicción .pkl
'''
async def preprocess_request(body: PredictPriceRequest) -> pd.DataFrame:
    # 1. Construir el dict base
    data = body.model_dump()
    data["Storage type"] = data.pop("Storage_type")

    # 2. Fecha actual y año de referencia
    today = datetime.now().date()
    ref_year = 2023
    current_year = today.year
    years_diff = current_year - ref_year
    data["request_date"] = str(today)

    # 3. Llamada al LLM con instrucciones de reproyección temporal
    client = AzureOpenAI(
        api_key    = require_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint = require_env("AZURE_OPENAI_ENDPOINT"),
        api_version    = require_env("AZURE_OPENAI_API_VERSION"),
    )

    system_msg = (
        f"Eres un experto en datos de laptops basados en un dataset de {ref_year}. "
        f"Hoy es {today} ({years_diff} años después). "
        "Debes mapear cualquier CPU, GPU, tamaño de pantalla, marca, modelo o Storage_type a su "
        "equivalente de 2023, **pero sólo** a UNA de las siguientes opciones:\n\n" +
        "\n".join(f"- **{k}**: {v}" for k, v in KNOWN_BY_PRICE_MODEL.items()) +
        "\n\nDevuélve **solo** un objeto JSON con estas claves: "
        "Status, Brand, Model, CPU, RAM, Storage, Storage type, GPU, Screen, Touch. "
        "Sin markdown ni texto adicional."
    )

    completion = client.chat.completions.create(
        model=require_env("AZURE_OPENAI_MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": json.dumps(data, ensure_ascii=False)},
        ]
    )

    raw = completion.choices[0].message.content.strip()
    # quitar fences si por error las incluye
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        mapped = json.loads(raw)
        ic(mapped)
    except JSONDecodeError:
        # fallback a los datos originales si no parsea
        mapped = data


    return pd.DataFrame([mapped])

'''
Valida que la imagen de la laptop enviada por endpoint corresponda al modelo y marca.
Si no, lanza un error de dominio.
'''
async def validate_laptop_name(image: UploadFile, brand: str, model: str) -> None:
    # Codifica la imagen a base64
    client = AzureOpenAI(
        api_key    = require_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint = require_env("AZURE_OPENAI_ENDPOINT"),
        api_version    = require_env("AZURE_OPENAI_API_VERSION"),
    )

    image_bytes = await image.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{image.content_type};base64,{b64}"
    ic(data_uri)

    # Llamada al LLM vision-enabled
    resp = client.chat.completions.create(
        model="gpt-4o",  # o tu despliegue vision
        messages=[
            {"role": "system", "content": (
                f"Eres un experto en visión por computadora. "
                f"Responde 'Sí' o 'No' estrictamente."
            )},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": f"¿Esta imagen corresponde a una laptop {brand} {model}? "
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{data_uri}"}
                }
            ]},
        ]
    )
    ic(resp)
    answer = resp.choices[0].message.content.lower()
    ic(answer)
    if "no" in answer or "incorrecta" in answer:
        raise InvalidLaptopError()
