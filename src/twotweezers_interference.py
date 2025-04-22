import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
# Define pixel size (4 µm)
pixel_size = 3.45e-6  # 4 microns per pixel

# Define image dimensions based on pixel size
image_width = 1920  # in pixels
image_height = 1200  # in pixels
image_width_microns = image_width * pixel_size  # 7.68 mm
image_height_microns = image_height * pixel_size  # 4.8 mm

# Create a blank 8-bit grayscale image
image = np.zeros((image_height, image_width), dtype=np.uint8)

# Define the 5x5 grid with each "spot" being 25x25 pixels
grid_size = 5  # Number of rows and columns in the grid
pixel_value = 255  # Intensity value for the "bright spots"
spot_size = 1  # Each spot is 25 pixels (100 µm)
spacing = 100 # Spacing between spots in pixels

# Starting positions for the grid (center the grid)
start_x = (image_width - (grid_size - 1) * spacing) // 2
start_y = (image_height - (grid_size - 1) * spacing) // 2

# Place the spots correctly
for i in range(grid_size):
    for j in range(grid_size):
        x_start = start_x + j * spacing
        y_start = start_y + i * spacing
        image[y_start, x_start] = pixel_value  # Place the bright pixel

# Normalize the image
target_im = image.astype(np.float32) / 255.0  # Normalize to range [0,1]

# === Find All Spot Positions in the Original Image ===
spot_positions = np.column_stack(np.where(target_im > 0))  # Find bright pixels (spots)
print(f"Original Spot Positions:\n {spot_positions}")


# Save the smoothed image
np.save(r"C:\Users\Yb\SLM\SLM\data\target_images\2x2tweezers.npy", target_im)

# Plot original and smoothed images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(target_im)
axs[0].set_title("Original Target Image")
axs[1].imshow(target_im, cmap='gray')
axs[1].set_title("Smoothed Target Image (σ=3)")
plt.xlim(930, 1000)  # Adjust x-axis limits to zoom in
plt.ylim(650, 550)  # Adjust y-axis limits to zoom in

plt.show()

print(f"Max value before smoothing: {np.max(target_im)}")


# Print image properties
print(f"Image width: {image_width_microns * 1e6} µm")
print(f"Image height: {image_height_microns * 1e6} µm")
print(f"Spot size: {spot_size * pixel_size * 1e6} µm")
print(f"Spacing: {spacing * pixel_size * 1e6} µm")
