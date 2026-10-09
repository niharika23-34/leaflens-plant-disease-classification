
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the LeafLens classification model."
    )
    parser.add_argument("--data", required=True,
                        help="Directory containing test class folders")
    parser.add_argument("--model", default="models/leaflens.keras")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {data_dir}")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first."
        )

    test_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(128, 128),
        batch_size=32,
        shuffle=False,
    )

    model = keras.models.load_model(model_path)
    class_names = test_ds.class_names

    true_labels = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])
    probabilities = model.predict(test_ds, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    if len(true_labels) != len(predicted_labels):
        raise ValueError("Prediction count does not match test labels.")

    if probabilities.shape[1] != len(class_names):
        raise ValueError("Test classes do not match model output classes.")

    print("\nTest Accuracy:",
          f"{accuracy_score(true_labels, predicted_labels):.4f}")

    print("\nClassification Report:\n")
    print(classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    ))

    print("Confusion Matrix:\n")
    print(confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    ))


if __name__ == "__main__":
    main()
