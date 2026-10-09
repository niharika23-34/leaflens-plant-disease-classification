
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    parser = argparse.ArgumentParser(
        description="Train the LeafLens plant leaf classifier."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "Dataset must contain train/ and validation/ folders."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(128, 128),
        batch_size=32,
        seed=42,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(128, 128),
        batch_size=32,
        shuffle=False,
    )

    class_names = train_ds.class_names

    if len(class_names) < 2:
        raise ValueError("At least two classes are required.")

    if class_names != val_ds.class_names:
        raise ValueError(
            "Training and validation class folders must match."
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "leaflens.keras")

    with open(
        output_dir / "classes.json", "w", encoding="utf-8"
    ) as file:
        json.dump(class_names, file, indent=2)

    print("Training completed.")
    print(f"Model saved in: {output_dir / 'leaflens.keras'}")
    print(f"Class labels saved in: {output_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
