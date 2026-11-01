import unittest

from inventory.store import Store


class StoreTests(unittest.TestCase):
    def test_add_and_remove(self):
        store = Store()
        store.add_item("bolt", 2, 0.1)
        self.assertEqual(store.remove_item("bolt").name, "bolt")

    def test_restock_increases_quantity(self):
        store = Store()
        store.add_item("bolt", 2, 0.1)
        store.restock("bolt", 3)
        self.assertEqual(store.items["bolt"].quantity, 5)


if __name__ == "__main__":
    unittest.main()
