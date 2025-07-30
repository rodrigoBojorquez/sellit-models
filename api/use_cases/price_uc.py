from fastapi.datastructures import UploadFile
from os import system
from api.common.errors import InvalidLaptopError
from api.common.constants import KNOWN_BY_DATASET
from api.contracts.price_contracts import PredictRequest
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
async def preprocess_request(body: PredictRequest) -> pd.DataFrame:
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
        "\n".join(f"- **{k}**: {v}" for k, v in KNOWN_BY_DATASET.items()) +
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