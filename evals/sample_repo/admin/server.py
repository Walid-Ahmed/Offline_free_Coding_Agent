"""Internal admin API, served separately from the public app."""
from fastapi import FastAPI

admin = FastAPI(title="Inventory Admin")


@admin.post("/restock/{name}")
def restock(name: str, amount: int):
    return {"name": name, "amount": amount}
