"""Data models for inventory items."""
from dataclasses import dataclass


@dataclass
class Item:
    name: str
    quantity: int
    unit_price: float

    @property
    def total_value(self):
        """Value of all units in stock."""
        return self.quantity * self.unit_price
