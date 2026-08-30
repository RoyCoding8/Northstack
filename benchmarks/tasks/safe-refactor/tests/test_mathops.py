import unittest

from mathops import concat, total


class TestMathops(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total([1, 2, 3]), 6)

    def test_total_empty(self):
        self.assertEqual(total([]), 0)

    def test_concat(self):
        self.assertEqual(concat(["a", "b"]), "ab")


if __name__ == "__main__":
    unittest.main()
