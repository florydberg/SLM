import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# ------------------ Parameters ------------------
frame_dir = r"C:\Users\Yb\SLM\SLM\data\captured_frames"
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.npy')])
num_peaks = 25
min_distance = 5
fps = 236  # frames per second

# ------------------ Peak Std Function ------------------
def intensity_std(u_int):
    coordinates_ccd = peak_local_max(
        u_int,
        min_distance=min_distance,
        num_peaks=num_peaks
    )

    peak_intensities = u_int[coordinates_ccd[:, 0], coordinates_ccd[:, 1]]

    mean_intensity = np.mean(peak_intensities)
    std_intensity = np.std(peak_intensities)
    relative_std = std_intensity / mean_intensity if mean_intensity > 0 else 1.0

    return relative_std, peak_intensities, coordinates_ccd

# ------------------ Loop Through Frames ------------------
stds_over_time = []
all_peak_intensities = []
i = 0
for frame_file in frame_files:
    frame_path = os.path.join(frame_dir, frame_file)
    frame = np.load(frame_path)

    rel_std, peak_vals, _ = intensity_std(frame)
    stds_over_time.append(rel_std)
    all_peak_intensities.append(peak_vals)
    print(i)
    i+=1

stds_over_time = np.array(stds_over_time)
all_peak_intensities = np.array(all_peak_intensities)  # shape: (num_frames, num_peaks)

# ------------------ Time Axis ------------------
times = np.arange(len(stds_over_time)) / fps  # seconds

# ------------------ Plot STD Over Time ------------------
plt.figure(figsize=(10, 4))
plt.plot(times, stds_over_time, label='Relative Std Dev')
plt.xlabel("Time (s)")
plt.ylabel("Relative Std Dev")
plt.title("Uniformity of Tweezer Intensities Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ Optional: Individual Peaks ------------------
plt.figure(figsize=(10, 6))
for i in range(num_peaks):
    peak = all_peak_intensities[:, i]
    peak_normalized = peak / np.max(peak)  # Normalize to [0, 1]
    plt.plot(times, peak_normalized, label=f"Peak {i+1}")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Peak Intensity")
plt.title("Normalized Peak Intensities Over Time")
#plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()
