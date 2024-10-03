from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from utils import predict

app = FastAPI()

class PredictReqArgs(BaseModel):
    ilce: str
    koy: str
    tarimSekli: str
    fosfor: float
    potasyum: float
    organikMadde: float
    ph: float
    kirec: float
    toplamTuz: float
    saturasyon: float

@app.post("/fruits/")
async def get_fruit_scores(reqArgs: PredictReqArgs):
    # This function would be where you handle the logic to score each fruit.
    # For demonstration, we're just returning a mock score.
    if not reqArgs:
        raise HTTPException(status_code=400, detail="No fruit data provided.")

    # Generating a mock score for each fruit
    result = predict(reqArgs.model_dump())
    return {"predictions": result}

# The server can be started using Uvicorn with the command:
# uvicorn filename:app --reload
