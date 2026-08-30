import unittest

from arith import add, div, mul


class TestArith(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)

    def test_mul(self):
        self.assertEqual(mul(3, 4), 12)

    def test_mul_zero(self):
        self.assertEqual(mul(5, 0), 0)

    def test_div(self):
        self.assertEqual(div(10, 4), 2.5)


if __name__ == "__main__":
    unittest.main()
