# ECG Arrhythmia Prediction

A machine learning project for classifying ECG heartbeat signals into five heartbeat categories using the MIT-BIH Arrhythmia Dataset.

## Project Overview

This project applies supervised learning to ECG heartbeat data and predicts five heartbeat classes:

- **N** — Normal beats
- **S** — Supraventricular ectopic beats
- **V** — Ventricular ectopic beats
- **F** — Fusion beats
- **Q** — Unknown or unclassified beats

The workflow includes ECG data exploration, class balancing through augmentation, model training with TensorFlow/Keras, and evaluation using precision, recall, F1 score, and a confusion matrix.

## Dataset

The project uses the MIT-BIH Arrhythmia Dataset.

- 109,446 total entries
- 188 columns
- The first 187 values represent ECG signal measurements
- The final value represents the heartbeat class

## Methodology

1. Load and combine the training and test CSV datasets
2. Separate ECG features and class labels
3. Visualize one ECG beat for each category
4. Apply data augmentation to help balance the classes
5. Create balanced test subsets
6. One-hot encode the target labels
7. Build a 1D convolutional neural network using Keras
8. Train the model
9. Evaluate predictions with classification metrics and a confusion matrix

## Data Augmentation

The project uses signal transformation functions such as:

- Stretching/resampling
- Amplitude adjustment
- Combined transformations

These transformations are used to create additional samples for underrepresented classes.

## Model

The neural network is built with TensorFlow/Keras and includes:

- Conv1D layers
- ReLU activations
- Residual/add connections
- MaxPooling1D layers
- Flatten layer
- Dense layers
- Softmax output for five-class prediction

The model is trained with categorical cross-entropy loss and the Adam optimizer.

## Evaluation

The model is evaluated using:

- Precision
- Recall
- F1 score
- Classification report
- Confusion matrix

The project report records an overall precision, recall, and F1 score of approximately **0.95** on a balanced evaluation set of 4,000 heartbeat samples.

### Reported class-level results

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| N | 0.84 | 0.99 | 0.91 |
| S | 1.00 | 0.85 | 0.92 |
| V | 0.96 | 0.96 | 0.96 |
| F | 0.98 | 0.96 | 0.97 |
| Q | 0.99 | 0.99 | 0.99 |

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## Repository Files

- `arrhythmia_prediction.py` — Python implementation
- `mitbih_train.csv` — training dataset
- `mitbih_test.csv` — test dataset
- `Report.docx` — written project report
- `Predicting-Arrhythmia.pptx` — project presentation

## Skills Demonstrated

Machine Learning • Deep Learning • ECG Signal Analysis • CNN • TensorFlow • Keras • Data Augmentation • Classification • Confusion Matrix • Precision • Recall • F1 Score

## Disclaimer

This project was created for educational and analytical purposes and is not intended for clinical diagnosis or medical decision-making.
