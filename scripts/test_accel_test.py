
from glob import glob
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from spectral_features import generate_features


# Path to the JSON files containing accelerometer data for testing
jsonPath="..//gestos_nano_esp32-export//testing"

# 1. Generate features and labels from the JSON files
test_input, test_output = generate_features(jsonPath)

# 2. Load the trained model from the file
model=keras.models.load_model('gesture_model.h5')

# 3. Evaluate the model on the test data
loss, accuracy = model.evaluate(x=test_input, y=test_output)





