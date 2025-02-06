from fastapi import FastAPI
from src.main.routes import router

app = FastAPI()

app.include_router(router, prefix="/api")
