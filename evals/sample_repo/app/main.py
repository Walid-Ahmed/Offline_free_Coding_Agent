"""Public inventory API."""
from fastapi import FastAPI

from inventory.store import Store

app = FastAPI(title="Inventory API")
store = Store()


@app.get("/items")
def list_items():
    return [item.name for item in store.items.values()]
