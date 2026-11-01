"""In-memory item store."""
from config import LOW_STOCK_THRESHOLD, MAX_ITEMS
from inventory.models import Item


class Store:
    def __init__(self):
        self.items = {}

    def add_item(self, name, quantity, unit_price):
        if len(self.items) >= MAX_ITEMS:
            raise ValueError("Store is full")
        self.items[name] = Item(name, quantity, unit_price)
        return self.items[name]

    def remove_item(self, name):
        """Remove an item and return it."""
        if name not in self.items:
            raise KeyError(f"Unknown item: {name}")
        return self.items.pop(name)

    def restock(self, name, amount):
        if amount <= 0:
            raise ValueError("Restock amount must be positive")
        self.items[name].quantity += amount

    def low_stock(self, threshold=LOW_STOCK_THRESHOLD):
        """Return items whose quantity is at or below the threshold."""
        return [item for item in self.items.values() if item.quantity < threshold]
