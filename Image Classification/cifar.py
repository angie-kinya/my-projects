import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Class names in CIFAR-10 - define them manually since the dataset does not include them
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Print data shapes - X_train and X_test contain 32x32x3 pixel images while y_train and y_test contain labels
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Visualize a few sample images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# Normaliza pixel values to range 0 to 1
X_train, X_test = X_train / 255.0, X_test / 255.0
print(f"Training data shape after normalization: {X_train.shape}, Testing data shape after normalization: {X_test.shape}")

# Build a Convolutional Neural Network Layer(CNN) model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(62, (3, 3), activation='relu'),
    layers.Flatten(), #Converts 2D matrix to 1D vector

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') #10 classes
])
#model summary
model.summary()

#Compile the model
model.compile(optimizer='adam', #adam is effective for CNNs and handles adaptive learning rates
              loss='sparse_categorical_crossentropy', #this is used dince the labels are integer coded
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot the training and validation accuracy
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

# Predict and Visualize model output
predictions = model.predict(X_test)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i])
    plt.title(f"True: {class_names[y_test[i][0]]}\nPredicted: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
plt.show()