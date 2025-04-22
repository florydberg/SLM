import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# ---------- Parameters ----------
image_dir = r"C:\Users\Yb\SLM\SLM\data\images"
file_list = sorted([f for f in os.listdir(image_dir) if f.startswith("tweezer_step_") and f.endswith(".npy")])
num_steps = len(file_list)
num_peaks = 5
min_distance = 10

# ---------- Helper: FFT Phase ----------
def get_fft_phase_and_intensity(hologram):
    fft = np.fft.fftshift(np.fft.fft2(hologram))
    intensity = np.abs(fft) ** 2
    phase = np.angle(fft)
    return intensity, phase

# ---------- Step 1: Get initial peak positions from first hologram ----------
first_frame = np.load(os.path.join(image_dir, file_list[0]))
fft_intensity, _ = get_fft_phase_and_intensity(first_frame)

coords = peak_local_max(fft_intensity, min_distance=min_distance, num_peaks=num_peaks)
print("Tracked coordinates (y, x):")
print(coords)

# ---------- Step 2: Track phase at those positions ----------
all_phases = []

for file in file_list:
    hologram = np.load(os.path.join(image_dir, file))
    _, phase = get_fft_phase_and_intensity(hologram)
    frame_phases = [phase[y, x] for y, x in coords]
    all_phases.append(frame_phases)

unwrap = all_phases
all_phases = np.array(all_phases)
# ---------- Step 3: Plot ----------
plt.figure(figsize=(10, 6))
for i in range(num_peaks):
    plt.plot(
        all_phases[:, i],
        linestyle=':',
        marker='o',
        markersize=4,
        label=f"Tweezer {i+1}"
    )

plt.xlabel("GSW phase mask")
plt.ylabel("Phase (radians)")
plt.title("Phase at 25 Tweezer Peaks Over Time")
plt.ylim(-10, 10)  # Set y-axis range to [0, 2π]

plt.legend(ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()

all_phases = np.unwrap(np.array(unwrap), axis=0)  # unwrap over time


# ---------- Step 3: Plot ----------
plt.figure(figsize=(10, 6))
for i in range(num_peaks):
    plt.plot(
        all_phases[:, i],
        linestyle=':',
        marker='o',
        markersize=4,
        label=f"Tweezer {i+1}"
    )

plt.xlabel("GSW phase mask")
plt.ylabel("Phase (radians)")
plt.ylim(-10, 10)  # Set y-axis range to [0, 2π]
plt.title("Phase at 25 Tweezer Peaks Over Time")
plt.legend(ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
