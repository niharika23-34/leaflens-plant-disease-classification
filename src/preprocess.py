
import argparse
from pathlib import Path

import cv2


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess_image(image_path, size=128):
    """Load and resize a plant leaf image."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError("Could not read the image file.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    image = image.astype("float32") / 255.0

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a plant leaf image."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be a positive integer")

    image = preprocess_image(args.input, args.size)

    print("Image preprocessing successful.")
    print(f"Output shape: {image.shape}")
    print(f"Pixel value range: {image.min():.3f} to {image.max():.3f}")


if __name__ == "__main__":
    main()

