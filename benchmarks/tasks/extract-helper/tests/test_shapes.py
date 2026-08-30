import unittest

from shapes import rectangle_area, triangle_area


class TestShapes(unittest.TestCase):
    def test_rectangle(self):
        self.assertEqual(rectangle_area(2, 3), 6)

    def test_triangle(self):
        self.assertEqual(triangle_area(4, 3), 6)

    def test_rectangle_rejects_negative(self):
        with self.assertRaises(ValueError):
            rectangle_area(-1, 3)

    def test_triangle_rejects_negative(self):
        with self.assertRaises(ValueError):
            triangle_area(3, -2)


if __name__ == "__main__":
    unittest.main()
