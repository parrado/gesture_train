import numpy as np
from tensorflow import keras
from spectral_features import generate_features
import matplotlib.pyplot as plt
from tensorflow import lite



# Path to the JSON files containing accelerometer data for training
jsonPath="..//dataset//training"

# 1. Generate features and labels from the JSON files
train_input, train_output = generate_features(jsonPath)



# 2. Build the model
model = keras.Sequential([
    keras.Input(shape=(27,)),  # Input layer with 27 features    
    keras.layers.Dense(10, activation='relu'), # Hidden layer with 10 neurons and 'relu' activation
    # Output layer with 2 neurons and 'softmax' activation
    keras.layers.Dense(2, activation='softmax')
])

# 3. Compile the model
# Use 'categorical_crossentropy' loss for multi-class problems with one-hot labels
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
history=model.fit(train_input, train_output, epochs=20, batch_size=16, validation_split=0.2)  # Train for 20 epochs with a batch size of 16 and 20% validation split

converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Assuming history is from model.fit()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# Plotting Accuracy
plt.figure(1)
plt.plot(acc)
plt.plot(val_acc)   
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')  # Add legend to differentiate between training and validation accuracy
plt.title('Accuracy')
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Accuracy')  # Add y-axis label
plt.grid()  # Add grid for better visibility




# Plotting Loss
plt.figure(2)
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')  # Add legend
plt.title('Loss')
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Loss')  # Add y-axis label
plt.grid()  # Add grid for better visibility    
plt.show()
print(len(loss))








