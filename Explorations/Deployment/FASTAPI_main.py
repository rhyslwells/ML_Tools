from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel
from typing import Optional

# Create a FastAPI app instance
app = FastAPI()

# Define a Pydantic model for data validation
class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    on_offer: Optional[bool] = None

# Root route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI example!"}

# Path parameter
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "query": q}

# Query parameter with constraints
@app.get("/search/")
async def search_items(limit: int = Query(10, ge=1, le=50), offset: int = 0):
    return {"limit": limit, "offset": offset}

# Create an item
@app.post("/items/")
async def create_item(item: Item):
    return {"message": "Item created successfully", "item": item}

# Update an item
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid item ID")
    return {"item_id": item_id, "updated_item": item}

# Delete an item
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid item ID")
    return {"message": f"Item {item_id} deleted successfully"}
