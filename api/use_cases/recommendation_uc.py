from openai import AzureOpenAI
from api.utils.require_env import require_env
from datetime import datetime


async def recommend_laptop(prompt: str):
    client = AzureOpenAI(
        api_key    = require_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint = require_env("AZURE_OPENAI_ENDPOINT"),
        api_version    = require_env("AZURE_OPENAI_API_VERSION"),
    )
    today = datetime.now().date()
    ref_year = 2023

    system_msg = (
        f"Eres un experto en recomendaciones de laptops basados en un dataset de {ref_year}. "
        f"Hoy es {today}. "
        "Debes recomendar un listado de 3 laptops basadas en las especificaciones dadas. "
        "Devuelve un texto sin formato adicional, que pueda ser interpolado directamente en una etiqueta HTML."
    )
    completion = client.chat.completions.create(
        model=require_env("AZURE_OPENAI_MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]
    )
    raw = completion.choices[0].message.content.strip()
    return raw