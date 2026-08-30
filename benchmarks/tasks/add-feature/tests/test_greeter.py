import unittest

from greeter import Greeter


class TestGreeter(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(Greeter().greet("Ada"), "Hello, Ada!")

    def test_farewell(self):
        self.assertEqual(Greeter().farewell("Ada"), "Goodbye, Ada!")

    def test_farewell_shout(self):
        self.assertEqual(Greeter().farewell("bob").endswith("!"), True)


if __name__ == "__main__":
    unittest.main()
