import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ------------------ Parameters ------------------
frame_dir = r"C:\Users\Yb\SLM\SLM\data\captured_frames"
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.npy')])
num_peaks = 25
min_distance = 5

# ------------------ Init Plot ------------------
plt.ion()
initial_frame = np.load(os.path.join(frame_dir, frame_files[0]))

fig, ax = plt.subplots(figsize=(10, 8))
img_plot = ax.imshow(initial_frame, cmap='gray')
colorbar = plt.colorbar(img_plot, ax=ax)
ax.set_title("Live Frame Viewer with Numbered Peaks")
ax.set_xlabel("X Pixels")
ax.set_ylabel("Y Pixels")
text_annotations = []

# ------------------ Track Peaks ------------------

# Get initial peak positions
reference_coords = peak_local_max(initial_frame, min_distance=min_distance, num_peaks=num_peaks)

# Loop through frames
for i, filename in enumerate(frame_files):
    frame_path = os.path.join(frame_dir, filename)
    frame = np.load(frame_path)

    # Detect peaks
    curr_coords = peak_local_max(frame, min_distance=min_distance, num_peaks=num_peaks)

    # Match peaks to reference using Hungarian algorithm (minimize distance)
    cost_matrix = cdist(reference_coords, curr_coords)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_coords = curr_coords[col_ind]

    # Update image
    img_plot.set_data(frame)
    img_plot.set_clim(vmin=frame.min(), vmax=frame.max())
    ax.set_title(f"Frame {i:04d} - {filename}")
    # Remove previous text annotations
    for t in text_annotations:
        t.remove()
    text_annotations = []

    # Draw numbered peaks
    for j, (y, x) in enumerate(matched_coords):
        txt = ax.text(x, y, str(j + 1), color='red', fontsize=8, ha='center', va='center', weight='bold')
        text_annotations.append(txt)

    plt.pause(0.01)

plt.ioff()
plt.show()