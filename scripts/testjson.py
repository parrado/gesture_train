import pandas as pd
from glob import glob
import os
import json
import matplotlib.pyplot as plt
from spectral_features import extract_accel_features
import numpy as np



jsonPath="..//dataset//training"
name=jsonPath+"//vertical.json"
vertical=[]
with open(name, 'r') as f:
    aux0 = json.load(f)   
    for item in aux0:
        if item['sensor'] == 'Accelerometer':
            vertical.append([float(item['x']), float(item['y']), float(item['z'])])

name=jsonPath+"//horizontal.json"
horizontal=[]
with open(name, 'r') as f:
    aux0 = json.load(f)   
    for item in aux0:
        if item['sensor'] == 'Accelerometer':
            horizontal.append([float(item['x']), float(item['y']), float(item['z'])])             

plt.figure(1)
plt.plot(vertical)
plt.title("Acceleration data from vertical gesture")
plt.xlabel("Index")     
plt.ylabel("Acceleration [m/s^2]")
plt.grid()  # Add grid for better visibility
plt.legend(["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"])


plt.figure(2)
plt.plot(horizontal)
plt.title("Acceleration data from horizontal gesture")
plt.xlabel("Index")     
plt.ylabel("Acceleration [m/s^2]")
plt.grid()  # Add grid for better visibility
plt.legend(["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"])



plt.show()








