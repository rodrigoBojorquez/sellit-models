from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
from api.routers.models import router as models_router
from pathlib import Path


BASE_DIR = Path(__file__).parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Loading models...")
    models_dir = BASE_DIR / "ai" / "models"

    # 2) Carga usando rutas absolutas
    app.state.models = {
        "gama_model":               joblib.load(models_dir / "sellit_gama_model.pkl"),
        "price_model":              joblib.load(models_dir / "sellit_price_model.pkl"),
        "priceperformance_model":   joblib.load(models_dir / "sellit_priceperformance_model.pkl"),
    }
    yield
    print("Shutting down: Unloading models...")
    app.state.models = None

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(models_router)