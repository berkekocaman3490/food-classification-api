from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from utils import predict_all

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictReqArgs(BaseModel):
    ilce: str
    potasyum: float
    fosfor: float
    ph: float

@app.post("/fruits/")
async def get_fruit_scores(reqArgs: PredictReqArgs):
    # This function would be where you handle the logic to score each fruit.
    # For demonstration, we're just returning a mock score.
    if not reqArgs:
        raise HTTPException(status_code=400, detail="No fruit data provided.")

    # Generating a mock score for each fruit
    result = predict_all(reqArgs.model_dump())
    print(result)
    return {"predictions": result}

# The server can be started using Uvicorn with the command:
# uvicorn main:app --reload
