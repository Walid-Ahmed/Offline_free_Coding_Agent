"""Pricing helpers."""
from config import DEFAULT_TAX_RATE


def apply_discount(price, percent):
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 100), 2)


def with_tax(price, rate=DEFAULT_TAX_RATE):
    return round(price * (1 + rate), 2)


def checkout_total(prices, discount_percent=0):
    """Sum prices, apply one discount, then add tax."""
    subtotal = apply_discount(sum(prices), discount_percent)
    return with_tax(subtotal)
