
import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess import preprocess_image


class TestPreprocess(unittest.TestCase):

    def test_valid_image(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "leaf.jpg"
            Image.new("RGB", (200, 100), color="green").save(image_path)

            result = preprocess_image(image_path)

            self.assertEqual(result.shape, (128, 128, 3))
            self.assertEqual(result.dtype, np.float32)
            self.assertGreaterEqual(result.min(), 0.0)
            self.assertLessEqual(result.max(), 1.0)

    def test_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image("this_image_does_not_exist.jpg")


if __name__ == "__main__":
    unittest.main()
