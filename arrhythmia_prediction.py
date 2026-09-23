# -*- coding: utf-8 -*-
"""
ECG Arrhythmia Prediction
"""

import math
import random
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import resample
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    label_ranking_average_precision_score,
    label_ranking_loss,
    coverage_error,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Conv1D,
    MaxPooling1D,
    Softmax,
    Add,
    Flatten,
    Activation,
)
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler


np.random.seed(42)


# Load data
df_train = pd.read_csv("mitbih_train.csv", header=None)
df_test = pd.read_csv("mitbih_test.csv", header=None)
df = pd.concat([df_train, df_test], axis=0)

M = df.values
X = M[:, :-1]
y = M[:, -1].astype(int)

del df, df_train, df_test, M


# Identify classes
C0 = np.argwhere(y == 0).flatten()
C1 = np.argwhere(y == 1).flatten()
C2 = np.argwhere(y == 2).flatten()
C3 = np.argwhere(y == 3).flatten()
C4 = np.argwhere(y == 4).flatten()


# Plot one ECG beat from each class
x = np.arange(0, 187) * 8 / 1000

plt.figure(figsize=(20, 12))
plt.plot(x, X[C0, :][0], label="Cat. N")
plt.plot(x, X[C1, :][0], label="Cat. S")
plt.plot(x, X[C2, :][0], label="Cat. V")
plt.plot(x, X[C3, :][0], label="Cat. F")
plt.plot(x, X[C4, :][0], label="Cat. Q")
plt.legend()
plt.title("1-beat ECG for every category", fontsize=20)
plt.ylabel("Amplitude", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)
plt.show()


# Data augmentation helpers
def stretch(signal):
    length = int(187 * (1 + (random.random() - 0.5) / 3))
    stretched = resample(signal, length)

    if length < 187:
        output = np.zeros(shape=(187,))
        output[:length] = stretched
    else:
        output = stretched[:187]

    return output


def amplify(signal):
    alpha = random.random() - 0.5
    factor = -alpha * signal + (1 + alpha)
    return signal * factor


def augment(signal):
    result = np.zeros(shape=(4, 187))

    for i in range(3):
        r = random.random()

        if r < 0.33:
            new_signal = stretch(signal)
        elif r < 0.66:
            new_signal = amplify(signal)
        else:
            new_signal = amplify(stretch(signal))

        result[i, :] = new_signal

    return result


# Example augmentation plot
plt.plot(X[0, :])
plt.plot(amplify(X[0, :]))
plt.plot(stretch(X[0, :]))
plt.show()


# Augment class F
result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)
class_labels = np.ones(shape=(result.shape[0],), dtype=int) * 3

X = np.vstack([X, result])
y = np.hstack([y, class_labels])


# Balanced test set
subC0 = np.random.choice(C0, 800)
subC1 = np.random.choice(C1, 800)
subC2 = np.random.choice(C2, 800)
subC3 = np.random.choice(C3, 800)
subC4 = np.random.choice(C4, 800)

X_test = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
y_test = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])

X_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
y_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)

del X, y


# Prepare inputs
X_train = np.expand_dims(X_train, 2)
X_test = np.expand_dims(X_test, 2)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

n_obs, feature, depth = X_train.shape
batch_size = 500

K.clear_session()


# Build model
inp = Input(shape=(feature, depth))
C = Conv1D(filters=32, kernel_size=5, strides=1)(inp)

C11 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(C)
A11 = Activation("relu")(C11)
C12 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(A11)
S11 = Add()([C12, C])
A12 = Activation("relu")(S11)
M11 = MaxPooling1D(pool_size=5, strides=2)(A12)

C21 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(M11)
A21 = Activation("relu")(C21)
C22 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(A21)
S21 = Add()([C22, M11])
A22 = Activation("relu")(S21)
M21 = MaxPooling1D(pool_size=5, strides=2)(A22)

C31 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(M21)
A31 = Activation("relu")(C31)
C32 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(A31)
S31 = Add()([C32, M21])
A32 = Activation("relu")(S31)
M31 = MaxPooling1D(pool_size=5, strides=2)(A32)

C41 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(M31)
A41 = Activation("relu")(C41)
C42 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(A41)
S41 = Add()([C42, M31])
A42 = Activation("relu")(S41)
M41 = MaxPooling1D(pool_size=5, strides=2)(A42)

C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(M41)
A51 = Activation("relu")(C51)
C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")(A51)
S51 = Add()([C52, M41])
A52 = Activation("relu")(S51)
M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

F1 = Flatten()(M51)
D1 = Dense(32)(F1)
A6 = Activation("relu")(D1)
D2 = Dense(32)(A6)
D3 = Dense(5)(D2)
A7 = Softmax()(D3)

model = Model(inputs=inp, outputs=A7)
model.summary()


# Learning-rate schedule
def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.75
    t = n_obs // (10000 * batch_size)
    return initial_lrate * math.exp(-k * t)


lrate = LearningRateScheduler(exp_decay)

adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(
    loss="categorical_crossentropy",
    optimizer=adam,
    metrics=["accuracy"],
)


# Train
history = model.fit(
    X_train,
    y_train,
    epochs=75,
    batch_size=batch_size,
    verbose=2,
    validation_data=(X_test, y_test),
    callbacks=[lrate],
)


# Evaluate
y_pred = model.predict(X_test, batch_size=1000)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(
    "Ranking-based average precision: {:.3f}".format(
        label_ranking_average_precision_score(y_test.toarray(), y_pred)
    )
)
print(
    "Ranking loss: {:.3f}".format(
        label_ranking_loss(y_test.toarray(), y_pred)
    )
)
print(
    "Coverage error: {:.3f}".format(
        coverage_error(y_test.toarray(), y_pred)
    )
)


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


cnf_matrix = confusion_matrix(
    y_test.argmax(axis=1),
    y_pred.argmax(axis=1),
)

plt.figure(figsize=(10, 10))
plot_confusion_matrix(
    cnf_matrix,
    classes=["N", "S", "V", "F", "Q"],
    title="Confusion matrix, without normalization",
)
plt.show()
