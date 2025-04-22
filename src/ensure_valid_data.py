import os
import numpy as np
import matplotlib.pyplot as plt

# Choose a specific file (replace with an actual basename)
basename = "tweezer_0173"

# Paths
base_path = r"C:\Users\Yb\SLM\SLM\data\target_images\WGS_CNN"
amp_path = os.path.join(base_path, "amplitude", f"{basename}_wgs_ampl.npy")
phase_path = os.path.join(base_path, "phase", f"{basename}_wgs_phase.npy")

# Load data
amplitude = np.load(amp_path)
phase = np.load(phase_path)

# === Plot & Save Amplitude Spectrum ===
plt.figure(figsize=(8, 6))
plt.imshow(amplitude, cmap="inferno", norm=plt.Normalize(vmin=0, vmax=np.max(amplitude)))
plt.colorbar(label="Amplitude")
plt.title("Amplitude Spectrum of FFT (Tweezer Pattern)")
plt.xlabel("Fourier X")
plt.ylabel("Fourier Y")
plt.tight_layout()
amp_png_path = os.path.join(base_path, f"{basename}_amplitude.png")
plt.savefig(amp_png_path)
plt.show()

# Save amplitude as .npy for consistency
amp_npy_out = os.path.join(base_path, f"{basename}_amplitude_copy.npy")
np.save(amp_npy_out, amplitude)

# === Plot & Save Phase Spectrum ===
plt.figure(figsize=(8, 6))
plt.imshow(phase, cmap="twilight", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
plt.colorbar(label="Phase (radians)")
plt.title("Phase Spectrum of FFT (Tweezer Pattern)")
plt.xlabel("Fourier X")
plt.ylabel("Fourier Y")
plt.tight_layout()
phase_png_path = os.path.join(base_path, f"{basename}_phase.png")
plt.savefig(phase_png_path)
plt.show()

# Save phase as .npy for consistency
phase_npy_out = os.path.join(base_path, f"{basename}_phase_copy.npy")
np.save(phase_npy_out, phase)