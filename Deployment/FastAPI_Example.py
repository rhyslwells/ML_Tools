from fastapi import FastAPI, Query, Path, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Create a FastAPI app instance
app = FastAPI()

# Define a Pydantic model for data validation
class Item(BaseModel):
    name: str  # Item name
    price: float  # Item price
    description: Optional[str] = None  # Optional description
    on_offer: Optional[bool] = None  # Optional flag for items on offer

class User(BaseModel):
    username: str  # Username of the user
    email: str  # Email of the user
    full_name: Optional[str] = None  # Optional full name
    items: Optional[List[Item]] = None  # Optional list of items owned by the user

# Store created, updated, and deleted items (in-memory database for demonstration)
created_items = []
updated_items = []
deleted_items = []

# Root route: A welcome endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the expanded FastAPI example!"}

# Path parameter: Access item details by ID with an optional query parameter
@app.get("/items/{item_id}")
async def read_item(item_id: int = Path(..., title="The ID of the item", ge=1), q: Optional[str] = None):
    """
    `item_id`: The ID of the item to retrieve (must be >= 1).
    `q`: An optional query parameter to filter or search.
    """
    return {"item_id": item_id, "query": q}

# Query parameters with constraints: Limit the number of results and specify offset
@app.get("/search/")
async def search_items(
    limit: int = Query(10, ge=1, le=50, description="Number of results to return (1-50)"),
    offset: int = Query(0, ge=0, description="Number of items to skip for pagination")
):
    return {"limit": limit, "offset": offset}

# Create an item: Validate data using Pydantic's BaseModel
@app.post("/items/")
async def create_item(item: Item):
    """
    Creates a new item using the `Item` model for validation.
    Returns a success message along with the item data.
    """
    created_items.append(item)
    return {"message": "Item created successfully", "item": item}

# Update an item: Update an item by its ID
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    """
    `item_id`: The ID of the item to update.
    `item`: The new data to update the item with.
    Raises an HTTPException if the ID is invalid.
    """
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid item ID")
    updated_items.append(item)
    return {"item_id": item_id, "updated_item": item}

# Delete an item: Delete an item by its ID
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """
    `item_id`: The ID of the item to delete.
    Raises an HTTPException if the ID is invalid.
    """
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid item ID")
    # Assume we have an `item` that gets deleted (just example; not retrieving actual items here)
    deleted_item = {"id": item_id, "name": f"Item-{item_id}"}  # Example of item deletion
    deleted_items.append(deleted_item)
    return {"message": f"Item {item_id} deleted successfully"}

# Nested models: Demonstrating the use of nested data structures
@app.post("/users/")
async def create_user(user: User):
    """
    Creates a user with optional items.
    Nested models allow for hierarchical data validation.
    """
    return {"message": "User created successfully", "user": user}

# Partial updates using Body: Update specific fields of an item
@app.patch("/items/{item_id}")
async def partial_update_item(
    item_id: int,
    price: Optional[float] = Body(None, description="The new price of the item"),
    on_offer: Optional[bool] = Body(None, description="Whether the item is on offer")
):
    """
    `item_id`: The ID of the item to partially update.
    Accepts optional fields for partial updates.
    """
    updates = {}
    if price is not None:
        updates["price"] = price
    if on_offer is not None:
        updates["on_offer"] = on_offer
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    return {"item_id": item_id, "updates": updates}

# Return static data: Useful for configuration or fixed responses
@app.get("/status/")
async def get_status():
    """
    Returns the API's status or any static information.
    """
    return {"status": "API is running", "version": "1.0.0"}

# Path operation to demonstrate returning a dictionary of key-value pairs
@app.get("/summary/")
async def get_summary():
    """
    Returns a dictionary with some computed or mock data.
    """
    # Adding actual counts of created, updated, and deleted items to the summary
    total_items = len(created_items) - len(deleted_items)  # Active items after deletions
    total_users = 5  # Example user count, static for this example
    recent_activity = "Item purchase"  # Mock activity
    return {
        "total_items": total_items,
        "total_users": total_users,
        "recent_activity": recent_activity,
        "created_items_count": len(created_items),
        "updated_items_count": len(updated_items),
        "deleted_items_count": len(deleted_items)
    }
