# ============================================================================
# HANDWRITTEN CHARACTER RECOGNITION USING CNN
# CodeAlpha Machine Learning Internship
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ============================================================================
# LOAD DATASET
# ============================================================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ============================================================================
# DISPLAY SAMPLE IMAGES
# ============================================================================

plt.figure(figsize=(10,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')

plt.show()

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

X_train = X_train.reshape(
    X_train.shape[0],
    28,
    28,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    28,
    28,
    1
)

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# ============================================================================
# BUILD CNN MODEL
# ============================================================================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        10,
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nCNN Model Summary:\n")
model.summary()

# ============================================================================
# TRAIN MODEL
# ============================================================================

history = model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ============================================================================
# EVALUATE MODEL
# ============================================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test_cat
)

print("\nTest Accuracy:")
print(round(accuracy, 4))

# ============================================================================
# PREDICTIONS
# ============================================================================

predictions = model.predict(X_test)

pred_classes = np.argmax(
    predictions,
    axis=1
)

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred_classes
    )
)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(
    y_test,
    pred_classes
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================================
# ACCURACY GRAPH
# ============================================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# ============================================================================
# PROJECT COMPLETED
# ============================================================================

print("\nProject Completed Successfully!")