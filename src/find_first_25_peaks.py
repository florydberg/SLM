import os
import numpy as np
from skimage.feature import peak_local_max

# ------------------ Parameters ------------------
frame_dir = r"C:\Users\Yb\SLM\SLM\data\captured_frames"
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.npy')])
num_peaks_required = 25
min_distance = 5

# ------------------ Find First Frame ------------------
for idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frame_dir, frame_file)
    frame = np.load(frame_path)

    peaks = peak_local_max(
        frame,
        min_distance=min_distance,
        num_peaks=num_peaks_required
    )

    if len(peaks) == num_peaks_required:
        print(f"✅ Found {num_peaks_required} peaks in file: {frame_file} (index {idx})")
        break
else:
    print(f"❌ No frame with exactly {num_peaks_required} peaks found.")
