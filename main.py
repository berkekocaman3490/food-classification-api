from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

class Fruit(BaseModel):
    name: str

@app.post("/fruits/")
async def get_fruit_scores(fruits: List[Fruit]):
    # This function would be where you handle the logic to score each fruit.
    # For demonstration, we're just returning a mock score.
    if not fruits:
        raise HTTPException(status_code=400, detail="No fruit data provided.")

    # Generating a mock score for each fruit
    results = [{'name': fruit.name, 'score': len(fruit.name) * 10} for fruit in fruits]
    return {"predictions": results}

# The server can be started using Uvicorn with the command:
# uvicorn filename:app --reload
