Implementations code 

MPORT LIBRARIES:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

LOAD DATASET (WISDM)
data = pd.read_csv("WISDM_ar_v1.1_raw.txt", header=None)

data.columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
data = data.dropna()

LABEL ENCODING
activity_map = {
    'Walking': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Downstairs': 5
}

data['activity'] = data['activity'].map(activity_map)

DATA STANDARDIZATION
scaler = StandardScaler()
data[['x','y','z']] = scaler.fit_transform(data[['x','y','z']])

WINDOWING (Frame Preparation)
Based on your doc:
Window size = 80
Step size = 40
window_size = 80
step_size = 40

X = []
y = []

for i in range(0, len(data) - window_size, step_size):
    xs = data['x'].values[i:i+window_size]
    ys = data['y'].values[i:i+window_size]
    zs = data['z'].values[i:i+window_size]
    
    label = data['activity'].mode()[0]
    
    X.append([xs, ys, zs])
    y.append(label)

X = np.array(X)
X = X.reshape(X.shape[0], 80, 3, 1)  # 2D CNN input
y = to_categorical(y, num_classes=6)






TRAIN-TEST SPLIT (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

CNN MODEL (BASED ON YOUR DOCUMENT)
model = Sequential()

# Reshape layer (as mentioned)
model.add(Reshape((80,3,1), input_shape=(80,3,1)))

# Convolution layers
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(6, activation='softmax'))






COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

TRAIN MODEL
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=80,
    validation_data=(X_test, y_test)
)

RESULTS (ACCURACY)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

Accuracy ≈ 95%
CONFUSION MATRIX
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
print(cm)


CLASSIFICATION REPORT
print(classification_report(y_true, y_pred_classes))

ACCURACY GRAPH

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

LOSS GRAPH

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()





