import os
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
# === Input directory ===
input_folder = r"C:\Users\Yb\SLM\SLM\data\target_images\WGS_CNN\Holograms"

# === Output directories ===
base_output = r"C:\Users\Yb\SLM\SLM\data\target_images\WGS_CNN"
amplitude_output = os.path.join(base_output, "amplitude")
phase_output = os.path.join(base_output, "phase")
peakmask_output = os.path.join(base_output, "peakmask")


# Create output folders if they don't exist
os.makedirs(amplitude_output, exist_ok=True)
os.makedirs(phase_output, exist_ok=True)
os.makedirs(peakmask_output, exist_ok=True)

# === Loop through all relevant .npy files ===
for fname in os.listdir(input_folder):
    if fname.endswith("_phasehologram.npy"):
        full_path = os.path.join(input_folder, fname)
        basename = fname.replace("_phasehologram.npy", "")

        # Load the phase hologram
        phase = np.load(full_path)

        # Apply FFT
        u_fft = fft2(np.exp(1j * phase))
        u_fft = fftshift(u_fft)

        # Extract Amplitude and normalize
        amplitude = np.abs(u_fft)
        amplitude /= np.max(amplitude)

        # Extract Phase
        phase_out = np.angle(u_fft)


        coordinates_last = peak_local_max(
            amplitude,
            min_distance=5,
            num_peaks=25
        )

        # Create a masked phase array with only the peak values
        masked_phase = np.zeros_like(phase_out)
        ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]
        masked_phase[ys, xs] = phase[ys, xs]

        peak_mask = np.zeros_like(amplitude)
        peak_mask[ys, xs] = 1.0

        new_amp = np.zeros_like(amplitude)
        new_amp[ys, xs] = amplitude[ys, xs]
        # === Plot final CCD image ===
        plt.figure(figsize=(10, 8))
        plt.imshow(new_amp, cmap='gray')
        plt.title("Final CCD Image")
        plt.xlabel("X Pixels")
        plt.ylabel("Y Pixels")
        plt.colorbar(label="Intensity")
        plt.tight_layout()

        plt.show()

        # === Save results ===
        amp_save_path = os.path.join(amplitude_output, f"{basename}_wgs_ampl.npy")
        phase_save_path = os.path.join(phase_output, f"{basename}_wgs_phase.npy")
        peak_mask_path = os.path.join(peakmask_output, f"{basename}_peakmask.npy")

        np.save(amp_save_path, new_amp)
        np.save(phase_save_path, masked_phase)
        np.save(peak_mask_path, peak_mask)
        
        print(f"[✓] Saved: {basename} -> amplitude + phase")

print("All files processed.")