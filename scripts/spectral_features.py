
import numpy as np
import json


# Function to generate features and labels from JSON files
def generate_features(jsonPath):

    # Empty lists to hold training data and labels
    input_data=[]
    output_data=[]

    name=jsonPath+"//vertical.json"
    with open(name, 'r') as f:
        aux0 = json.load(f)
        accel=[]
        for item in aux0:
            if item['sensor'] == 'Accelerometer':
                accel.append([float(item['x']), float(item['y']), float(item['z'])])        
        feature=extract_accel_features(accel)
        input_data.extend(feature)  # Append features to the training input
        output_data.extend([[1, 0]] * len(feature))  # Append labels for vertical gesture

    name=jsonPath+"//horizontal.json"
    with open(name, 'r') as f:
        aux0 = json.load(f)
        accel=[]
        for item in aux0:
            if item['sensor'] == 'Accelerometer':
                accel.append([float(item['x']), float(item['y']), float(item['z'])])        
        feature=extract_accel_features(accel)
        input_data.extend(feature)  # Append features to the training input
        output_data.extend([[0, 1]] * len(feature))  # Append labels for horizontal gesture
    


    # Convert lists to numpy arrays
    output_data=np.array(output_data)
    input_data=np.array(input_data)

    return input_data, output_data


# Function to extract features from accelerometer data
def extract_accel_features(signal):
    """
    Extracts 27 features per window from accelerometer signal.
    Signal shape expected: (N_samples, 27)
    """
    data = np.array(signal)
    n_samples = data.shape[0]
    
    window_size = 168
    stride = 17
    frame_size = 16
    frame_overlap = 8  # 50% of 16
    
    all_window_features = []
    
    # Iterate through the signal using the defined window and stride
    for start in range(0, n_samples - window_size + 1, stride):
        window = data[start : start + window_size]  # Shape: (168, 3)
        window_features = []
        
        # Process each axis (ax, ay, az) independently
        for axis in range(3):
            axis_data = window[:, axis]
            
            # 1. Compute RMS for the window
            rms = np.sqrt(np.mean(axis_data**2))
            
            # 2. Slice window into frames of 16 with 50% overlap
            frames = []
            for f_start in range(0, window_size - frame_size + 1, frame_overlap):
                frames.append(axis_data[f_start : f_start + frame_size])
            
            frames = np.array(frames) # Shape: (Number of frames, 16)
            
            # 3. Compute FFT for each frame
            # rfft returns n/2 + 1 bins. For n=16, it returns 9 bins (DC + 8 bins)
            ffts = np.abs(np.fft.rfft(frames, n=frame_size))
            
            # 4. Remove DC component (first bin) and keep the remaining 8 bins
            # Shape becomes (Number of frames, 8)
            spectral_bins = ffts[:, 1:]
            
            # 5. For each frequency bin, keep the largest magnitude across all frames
            max_spectrum = np.max(spectral_bins, axis=0)
            
            # 6. Compute log of the spectrum (adding small epsilon to avoid log(0))
            log_spectrum = np.log(max_spectrum + 1e-9)
            
            # Combine RMS (1) + Log Spectrum (8) = 9 features per axis
            window_features.extend([rms] + log_spectrum.tolist())
            
        all_window_features.append(window_features)
        
    return all_window_features
