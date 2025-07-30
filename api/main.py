from fastapi.responses import JSONResponse
from dotenv.main import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
from api.routers.models import router as models_router
from pathlib import Path
from fastapi import Request
from api.common import Error, ValidationError, NotFoundError, ConflictError

BASE_DIR = Path(__file__).parent
ERROR_STATUS_MAP = {
    ValidationError: 422,
    NotFoundError:         404,
    ConflictError:         409,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    print("Starting up: Loading models...")
    models_dir = BASE_DIR / "models"

    # 2) Carga usando rutas absolutas
    app.state.models = {
        "gama_model": joblib.load(models_dir / "sellit_gama_model.pkl"),
        "price_model": joblib.load(models_dir / "sellit_price_model.pkl"),
        "priceperformance_model": joblib.load(models_dir / "sellit_priceperformance_model.pkl"),
    }
    yield
    print("Shutting down: Unloading models...")
    app.state.models = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Error)
async def error_handler(request: Request, exc: Error):
    status_code = next(
        (code for exc_type, code in ERROR_STATUS_MAP.items() if isinstance(exc, exc_type)),
        500
    )
    return JSONResponse(status_code=status_code, content={"detail": exc.message})


app.include_router(models_router)