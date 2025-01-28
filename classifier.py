import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

# Load dataset
data = tf.keras.utils.image_dataset_from_directory('dataset')

# Preprocess data (normalize pixel values)
data = data.map(lambda x, y: (x / 255.0, y))

# Split dataset
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
validate = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])

model.summary()

# Train model
logdir = 'logs'
callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

trained_model = model.fit(train, epochs=20, validation_data=validate, callbacks=[callback])

# Plot Loss
plt.figure()
plt.plot(trained_model.history['loss'], label='Loss', color='teal')
plt.plot(trained_model.history['val_loss'], label='Validation Loss', color='orange')
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(trained_model.history['accuracy'], label='Accuracy', color='teal')
plt.plot(trained_model.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.legend()
plt.show()

# Evaluate model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

print(f'Precision: {precision.result().numpy()}, Recall: {recall.result().numpy()}, Accuracy: {accuracy.result().numpy()}')

# Load & predict on a single image
img = cv2.cvtColor(cv2.imread('animal.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))

print("It is a human!" if yhat > 0.5 else "It is an animal!")

# Saving the model.
model.save(os.path.join('model', 'img_classifier.h5'))
