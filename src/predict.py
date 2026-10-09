
import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from PIL import Image, UnidentifiedImageError


def predict_image(image_path, model_path, classes_path):
    image_path = Path(image_path)
    model_path = Path(model_path)
    classes_path = Path(classes_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Train first."
        )

    if not classes_path.is_file():
        raise FileNotFoundError(
            f"Class labels not found: {classes_path}. Train first."
        )

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((128, 128))
            image = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The input is not a valid image.") from exc

    image = np.expand_dims(image, axis=0)

    model = keras.models.load_model(model_path)

    with open(classes_path, encoding="utf-8") as file:
        class_names = json.load(file)

    probabilities = model.predict(image, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError("Model output and class labels do not match.")

    best_index = int(np.argmax(probabilities))

    return class_names[best_index], float(probabilities[best_index])


def main():
    parser = argparse.ArgumentParser(
        description="Predict the class of a plant leaf image."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/leaflens.keras")
    parser.add_argument("--classes", default="models/classes.json")
    args = parser.parse_args()

    label, confidence = predict_image(
        args.image, args.model, args.classes
    )

    print(f"Predicted class: {label}")
    print(f"Model confidence: {confidence:.2%}")
    print("Note: This prediction is not a confirmed plant diagnosis.")


if __name__ == "__main__":
    main()
