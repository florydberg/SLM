import numpy as np
import matplotlib.pyplot as plt

# Load the frame
frame_path = r"C:\Users\Yb\SLM\SLM\data\captured_frames\frame_0300.npy"
frame = np.load(frame_path)

# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(frame, cmap='gray')
plt.title("Captured Frame 0000")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")
plt.colorbar(label='Intensity')
plt.tight_layout()

