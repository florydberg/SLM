import numpy as np
import matplotlib.pyplot as plt
import os 

def generate_axis_safe_tweezers(image_width, image_height, box_size, min_distance, num_tweezers, intensity=255):
    image = np.zeros((image_height, image_width), dtype=np.uint8)
    forbidden = np.zeros_like(image, dtype=bool)
    coords = []

    center_x = image_width // 2
    center_y = image_height // 2
    half_box = box_size // 2

    xmin = center_x - half_box
    xmax = center_x + half_box
    ymin = center_y - half_box
    ymax = center_y + half_box

    attempts = 0
    max_attempts = 10000

    while len(coords) < num_tweezers and attempts < max_attempts:
        attempts += 1
        x = np.random.randint(xmin, xmax)
        y = np.random.randint(ymin, ymax)

        if not forbidden[y, x]:
            # Save tweezer
            coords.append((y, x))
            image[y, x] = intensity

            # Mark a square area as forbidden (± min_distance in x/y)
            x_min = max(0, x - min_distance)
            x_max = min(image.shape[1], x + min_distance + 1)
            y_min = max(0, y - min_distance)
            y_max = min(image.shape[0], y + min_distance + 1)
            forbidden[y_min:y_max, x_min:x_max] = True

    if len(coords) < num_tweezers:
        print(f"⚠️ Warning: Only placed {len(coords)} tweezers out of {num_tweezers}")

    return image.astype(np.float32) / 255.0, coords
# Parameters
image_width = 1920
image_height = 1200
box_size = 800
min_distance = 30
num_tweezers = 25


### SINGLE IMAGE#####
# # Random number of tweezers
# num_tweezers = np.random.randint(1, 201)
# print(f"Placing {num_tweezers} tweezers")

# # Generate
# target_im, coords = generate_axis_safe_tweezers(
#     image_width, image_height, box_size, min_distance, 25
# )

# # Plot
# plt.figure(figsize=(10, 6))
# plt.imshow(target_im, cmap='gray')
# plt.scatter([x for y, x in coords], [y for y, x in coords], color='red', marker='x')
# plt.title("Tweezer Placement (Strict 30px in X & Y)")
# plt.grid(True)
# plt.show()

# Save
#np.save(r"C:\Users\Yb\SLM\SLM\data\target_images\CNN\random_spaced_tweezers.npy", target_im)

### SINGLE IMAGE#####




# output_folder = r"C:\Users\Yb\SLM\SLM\data\target_images\CNN"

# os.makedirs(output_folder, exist_ok=True)

# # === Generate and Save ===  #### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS
# for i in range(3000):
#     num_tweezers = np.random.randint(1, 201)
#     target_im, coords = generate_axis_safe_tweezers(
#         image_width, image_height, box_size, min_distance, num_tweezers
#     )

#     filename = f"tweezer_{i:04d}_N{num_tweezers}.npy"
#     path = os.path.join(output_folder, filename)
#     np.save(path, target_im)

#     if i % 100 == 0:
#         print(f"Saved {i} / 3000")

# print("✅ Done generating 3000 random tweezer images.")
#### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS#### RANDOM NUMBER OF TWEEZERS




output_folder = r"C:\Users\Yb\SLM\SLM\data\target_images\CNN"

os.makedirs(output_folder, exist_ok=True)

# === Generate and Save ===  #### RANDOM NUMBER OF TWEEZERS
for i in range(3000):
    target_im, coords = generate_axis_safe_tweezers(
        image_width, image_height, box_size, min_distance, num_tweezers
    )

    filename = f"tweezer_{i:04d}.npy"
    path = os.path.join(output_folder, filename)
    np.save(path, target_im)

    if i % 100 == 0:
        print(f"Saved {i} / 3000")

print("✅ Done generating 3000 random tweezer images.")
