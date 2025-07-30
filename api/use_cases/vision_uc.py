from fastapi.datastructures import UploadFile
from os import system
from api.common.errors import InvalidLaptopError
from api.common.constants import KNOWN_BY_DATASET
from api.contracts.price_contracts import PredictRequest
import pandas as pd
from openai import AzureOpenAI
from api.utils.require_env import require_env
import json
from datetime import datetime
import base64
from icecream import ic

'''
Valida que la imagen de la laptop enviada por endpoint corresponda al modelo y marca.
Si no, lanza un error de dominio.
'''
async def validate_laptop_name(image: UploadFile, description: str) -> None:
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
                    "text": f"¿Esta imagen corresponde a una laptop con la siguiente descripción: {description}?"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{data_uri}"}
                }
            ]},
        ]
    )
    answer = resp.choices[0].message.content.lower()
    ic(answer)
    if "no" in answer or "incorrecta" in answer:
        raise InvalidLaptopError()
