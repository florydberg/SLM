import os
import numpy as np
from skimage.feature import peak_local_max

# Define the input directory containing the phase files
input_dir = r"C:\Users\Yb\SLM\SLM\data\target_images\WGS_CNN\phase"

# Parameters for peak detection
min_distance = 5
num_peaks = 25

# Process each .npy file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith("_wgs_phase.npy"):
        filepath = os.path.join(input_dir, filename)

        # Load the phase data
        phase = np.load(filepath)

        # Apply peak detection to get coordinates
        w = np.abs(np.exp(1j * phase))
        coordinates_last = peak_local_max(
            w,
            min_distance=min_distance,
            num_peaks=num_peaks
        )

        # Create a masked phase array with only the peak values
        masked_phase = np.zeros_like(phase)
        ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]
        masked_phase[ys, xs] = phase[ys, xs]

        # Save the updated masked phase back to the same file
        np.save(filepath, masked_phase)

"Updated all matching phase files with masked values at peak coordinates."


