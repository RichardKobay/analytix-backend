from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.main.data_processing import process_data

router = APIRouter()


@router.get("/user-tweets/{username}")
def user_tweets(username: str) -> JSONResponse:
    data = process_data("", "", "")
    return JSONResponse(content=data)
