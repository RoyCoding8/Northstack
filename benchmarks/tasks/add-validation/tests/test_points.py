import unittest

from points import distance


class TestDistance(unittest.TestCase):
    def test_simple(self):
        self.assertAlmostEqual(distance(0, 0, 3, 4), 5.0)

    def test_rejects_non_numeric_x1(self):
        with self.assertRaises(ValueError):
            distance("a", 0, 3, 4)

    def test_rejects_non_numeric_y2(self):
        with self.assertRaises(ValueError):
            distance(0, 0, 3, None)

    def test_accepts_ints_and_floats(self):
        self.assertAlmostEqual(distance(0, 0, 1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
