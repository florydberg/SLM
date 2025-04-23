import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Directory containing frames
frame_dir = r"C:\Users\Yb\SLM\SLM\data\captured_frames"

# Get all frame files sorted
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.npy')])

# Initialize the plot
plt.ion()  # Turn on interactive mode
# Load the first frame to initialize properly
initial_frame = np.load(os.path.join(frame_dir, frame_files[0]))

fig, ax = plt.subplots(figsize=(10, 8))
img_plot = ax.imshow(initial_frame, cmap='gray')
colorbar = plt.colorbar(img_plot, ax=ax)
ax.set_title("Live Frame Viewer")
ax.set_xlabel("X Pixels")
ax.set_ylabel("Y Pixels")

# Loop through frames
for i, filename in enumerate(frame_files):
    frame_path = os.path.join(frame_dir, filename)
    frame = np.load(frame_path)

    img_plot.set_data(frame)
    img_plot.set_clim(vmin=frame.min(), vmax=frame.max())  # auto scale contrast
    ax.set_title(f"Frame {i:04d} - {filename}")
    plt.pause(.01)  # 1-second delay

plt.ioff()
plt.show()
