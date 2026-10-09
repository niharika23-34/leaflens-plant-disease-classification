# LeafLens: Plant Leaf Disease Classification Using Deep Learning

## 1. Project Overview

LeafLens is a computer vision project that uses a Convolutional Neural Network (CNN) to classify plant leaf images into healthy or disease-related categories. It takes a leaf image as input and predicts its most likely class.

The project uses Python, TensorFlow, OpenCV, NumPy, and Pillow.

## 2. Objectives

- Classify plant leaf images into 38 categories.
- Build and train a CNN for image classification.
- Preprocess images by resizing and normalizing pixel values.
- Evaluate the trained model using test data.
- Predict the class of an individual leaf image.

## 3. Dataset

The project uses the PlantVillage dataset, specifically its color-image collection.

Dataset source: https://github.com/spMohanty/PlantVillage-Dataset

The dataset contains 38 classes covering several crops, including apples, corn, grapes, potatoes, and tomatoes.

The images were divided into training, validation, and testing sets.

| Dataset split | Number of images |
|---|---:|
| Training | 37,997 |
| Validation | 8,129 |
| Testing | 8,179 |
| Total | 54,305 |

## 4. Model Architecture

A Convolutional Neural Network (CNN) was implemented using TensorFlow and Keras.

The architecture consists of:

1. Input layer for 128 × 128 RGB images
2. Rescaling layer
3. Convolutional layer with 16 filters and ReLU activation
4. Max-pooling layer
5. Convolutional layer with 32 filters and ReLU activation
6. Max-pooling layer
7. Convolutional layer with 64 filters and ReLU activation
8. Global average pooling layer
9. Dense layer with 64 units and ReLU activation
10. Dropout layer with a rate of 0.3
11. Output layer with 38 units and softmax activation

The model contains 30,214 parameters and was trained for 3 epochs using the Adam optimizer and sparse categorical cross-entropy loss.

## 5. Project Structure

```text
leaflens-plant-disease-classification/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
│   └── test_preprocess.py
├── requirements.txt
└── README.md
```

The dataset and trained model are generated or stored separately and are not included in this source-code listing.

## 6. Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Download the PlantVillage dataset and arrange the selected images into the following directory structure:

```text
data/
├── train/
│   ├── class_1/
│   └── class_2/
├── validation/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_1/
    └── class_2/
```

Use the actual class folder names from the dataset. The training and validation directories must contain matching class folders.

## 7. Training the Model

Run the following command from the project root:

```bash
python src/train.py --data data --output models --epochs 3
```

After training, the script saves the trained model as `models/leaflens.keras` and the class labels as `models/classes.json`.

## 8. Evaluating the Model

Run:

```bash
python src/evaluate.py --data data/test --model models/leaflens.keras
```

The evaluation script reports test accuracy, precision, recall, F1-score, and a confusion matrix.

## 9. Predicting a Leaf Image

Run:

```bash
python src/predict.py --image path/to/leaf.jpg --model models/leaflens.keras --classes models/classes.json
```

Replace `path/to/leaf.jpg` with the path to your image.

## 10. Results

The model was trained for three epochs.

| Metric | Result |
|---|---:|
| Training accuracy, epoch 3 | 54.36% |
| Validation accuracy, epoch 3 | 59.68% |
| Test accuracy | 59.11% |
| Weighted F1-score | 0.58 |
| Macro F1-score | 0.45 |

A sample test image belonging to the `Tomato___healthy` class was correctly classified as `Tomato___healthy`, with a model confidence score of 80.89%.

## 11. Limitations

- The model was trained for only three epochs and can be improved.
- Classification performance varies across the 38 categories.
- The random image-level split may place related images in different subsets, potentially affecting evaluation.
- Performance on photographs taken in real-world conditions may differ from performance on PlantVillage images.
- The prediction is a model estimate, not a confirmed plant-health diagnosis.

## 12. Future Improvements

- Train for more epochs and tune hyperparameters.
- Experiment with transfer learning using MobileNetV2 or EfficientNet.
- Improve performance on classes with low recall.
- Investigate duplicate or closely related images across data splits.
- Build a simple web interface for image uploads and predictions.

## 13. Conclusion

LeafLens demonstrates an end-to-end deep-learning workflow for plant leaf image classification, including preprocessing, CNN training, model evaluation, and individual-image prediction. The current model achieves 59.11% test accuracy and provides a baseline for future improvements.
