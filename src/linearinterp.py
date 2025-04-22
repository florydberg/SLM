import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy.fft import fft2, fftshift
from skimage.feature import peak_local_max
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os 
from scipy import fft as sfft

# Load the .npy file
file_path = r"C:\Users\Yb\SLM\SLM\data\images\tweezer_step_0300.npy"  # Adjust if needed
phase_pattern_final = np.load(file_path)
file_path = r"C:\Users\Yb\SLM\SLM\data\images\tweezer_step_0000.npy"  # Adjust if needed
phase_pattern_initial = np.load(file_path)

# Apply FFT to get the tweezer pattern
u_fft_final = fft2(np.exp(1j * phase_pattern_final))  # Correctly apply FFT to the phase pattern
u_fft_final = fftshift(u_fft_final)

# Extract Amplitude and Phase
amplitude_final = np.abs(u_fft_final)  # Intensity of tweezers in Fourier space
phase_final = np.angle(u_fft_final)  # Phase information of tweezers

# Apply FFT to get the tweezer pattern
u_fft_initial = fft2(np.exp(1j * phase_pattern_initial))  # Correctly apply FFT to the phase pattern
u_fft_initial = fftshift(u_fft_initial)

# Extract Amplitude and Phase
amplitude_initial = np.abs(u_fft_initial)  # Intensity of tweezers in Fourier space
phase_initial = np.angle(u_fft_initial)  # Phase information of tweezers

coordinates_initial = peak_local_max(
    np.abs(u_fft_initial),
    min_distance=5,
    num_peaks=25
)

# plt.figure(figsize=(8, 6))
# plt.imshow(amplitude_final, cmap="inferno", norm=plt.Normalize(vmin=0, vmax=np.max(amplitude_final)))
# plt.colorbar(label="Amplitude")
# plt.title("Amplitude Spectrum of FFT (Tweezer Pattern)")
# plt.xlabel("Fourier X")
# plt.ylabel("Fourier Y")
# plt.show()

# # Plot Phase Spectrum
# plt.figure(figsize=(8, 6))
# plt.imshow(phase_final, cmap="twilight", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
# plt.colorbar(label="Phase (radians)")
# plt.title("Phase Spectrum of FFT (Tweezer Pattern)")
# plt.xlabel("Fourier X")
# plt.ylabel("Fourier Y")
# plt.show()


coordinates_final = peak_local_max(
    np.abs(u_fft_final),
    min_distance=5,
    num_peaks=25
)

# # Plot the magnitude spectrum of the FFT result
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.imshow(np.abs(u_fft_final), cmap="inferno", norm=plt.Normalize(vmin=0, vmax=np.max(np.abs(u_fft_final))))
# ax.set_title("Magnitude Spectrum of FFT (Tweezer Pattern)")
# ax.set_xlabel("Fourier X")
# ax.set_ylabel("Fourier Y")
# plt.colorbar(ax.imshow(np.abs(u_fft_final), cmap="inferno"), label="Magnitude")

# # Add squares around detected peaks
# for coord in coordinates_final:
#     rect = patches.Rectangle(
#         (coord[1] - 5, coord[0] - 5), 10, 10, linewidth=1.5, edgecolor='cyan', facecolor='none'
#     )
#     ax.add_patch(rect)

# plt.show()

# Extract phase and amplitude at detected tweezer positions
tweezer_amplitudes = amplitude_final[coordinates_final[:, 0], coordinates_final[:, 1]]  # Get amplitudes at tweezer locations
tweezer_phases = phase_final[coordinates_final[:, 0], coordinates_final[:, 1]]  # Get phases at tweezer locations

# Print extracted values
for i, (coord, amp, ph) in enumerate(zip(coordinates_final, tweezer_amplitudes, tweezer_phases)):
    print(f"Tweezer {i+1}: Position {coord}, Amplitude {amp:.4f}, Phase {ph:.4f} rad")





from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

dist_matrix = cdist(coordinates_initial, coordinates_final)
row_ind, col_ind = linear_sum_assignment(dist_matrix)

# Now: coords_0[i] maps to coords_1[col_ind[i]]

import matplotlib.pyplot as plt

# Make sure coords_0 and coords_1 are NumPy arrays
# coords_0 = np.array(coordinates_initial)
# coords_1 = np.array(coordinates_final)

# # Plot setup
# plt.figure(figsize=(8, 8))
# plt.scatter(coords_0[:, 1], coords_0[:, 0], c='blue', marker='o', label='Initial Tweezers')
# plt.scatter(coords_1[:, 1], coords_1[:, 0], c='red', marker='x', label='Final Tweezers')

# # Draw arrows between matched pairs
# for i, j in zip(row_ind, col_ind):
#     y0, x0 = coords_0[i]
#     y1, x1 = coords_1[j]
#     plt.arrow(x0, y0, x1 - x0, y1 - y0,
#               color='gray', alpha=0.6, width=0.3, head_width=5, length_includes_head=True)

# plt.gca().invert_yaxis()
# plt.title("Tweezer Mapping: Initial → Final")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()


N = 20
interpolated_frames = []
coords_0=coordinates_initial
coords_1=coordinates_final 

# Enforce phase continuity: use phi0 for both start and end
for i in range(len(row_ind)):
    y0, x0 = coords_0[row_ind[i]]
    y1, x1 = coords_1[col_ind[i]]

    # Set the final phase to be equal to the initial phase
    phase_final[y1, x1] = phase_initial[y0, x0]


# # Plot Amplitude Spectrum
# plt.figure(figsize=(8, 6))
# plt.imshow(amplitude_final, cmap="inferno", norm=plt.Normalize(vmin=0, vmax=np.max(amplitude_final)))
# plt.colorbar(label="Amplitude")
# plt.title("Amplitude Spectrum of FFT (Tweezer Pattern) final")
# plt.xlabel("Fourier X")
# plt.ylabel("Fourier Y")

# plt.figure(figsize=(8, 6))
# plt.imshow(amplitude_initial, cmap="inferno", norm=plt.Normalize(vmin=0, vmax=np.max(amplitude_final)))
# plt.colorbar(label="Amplitude")
# plt.title("Amplitude Spectrum of FFT (Tweezer Pattern) initial")
# plt.xlabel("Fourier X")
# plt.ylabel("Fourier Y")

# Plot Phase Spectrum
plt.figure(figsize=(8, 6))
plt.imshow(phase_final, cmap="twilight", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
plt.colorbar(label="Phase (radians)")
plt.title("Phase Spectrum of FFT (Tweezer Pattern) final")
plt.xlabel("Fourier X")
plt.ylabel("Fourier Y")


# Plot Phase Spectrum
plt.figure(figsize=(8, 6))
plt.imshow(phase_initial, cmap="twilight", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
plt.colorbar(label="Phase (radians)")
plt.title("Phase Spectrum of FFT (Tweezer Pattern) initial")
plt.xlabel("Fourier X")
plt.ylabel("Fourier Y")
plt.show()




U0 = np.zeros_like(phase_initial, dtype=complex)  # Just used for shape
interpolated_frames = []
N = 20  # Number of interpolated frames

for step in range(N):
    alpha = step / (N - 1)  # Linear interpolation factor
    U_interp = np.zeros_like(U0, dtype=complex)

    for i in range(len(row_ind)):
        # Initial and final positions
        y0, x0 = coords_0[row_ind[i]]
        y1, x1 = coords_1[col_ind[i]]

        # Interpolated position
        y = int(round((1 - alpha) * y0 + alpha * y1))
        x = int(round((1 - alpha) * x0 + alpha * x1))

        # Interpolated amplitude
        A0 = amplitude_initial[y0, x0]
        A1 = amplitude_final[y1, x1]
        A_interp = (1 - alpha) * A0 + alpha * A1

        # Phase (already forced to be the same in final)
        phi_interp = phase_initial[y0, x0]

        # Assign complex field
        U_interp[y, x] = A_interp * np.exp(1j * phi_interp)

    interpolated_frames.append(U_interp)

step_idx = 19
amp = np.abs(interpolated_frames[step_idx])
phase = np.angle(interpolated_frames[step_idx])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(amp, cmap="inferno")
plt.title(f"Interpolated Amplitude (step {step_idx})")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
plt.title(f"Interpolated Phase (step {step_idx})")
plt.colorbar()
plt.tight_layout()
plt.show()


output_dir = r"C:\Users\Yb\SLM\SLM\data\interpolated_phase_patterns"
os.makedirs(output_dir, exist_ok=True)

for idx, U in enumerate(interpolated_frames):
    # Reconstruct full complex field from amplitude and phase
    u = sfft.ifftshift(U)
    u = sfft.ifft2(u)
        
    

        # The phase that we need to imprint by the SLM is:
    phase = np.angle(u)
    # Convert phase from [-π, π] → [0, 2π] → [0, 255]
    phase_slm = np.flip(np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8), axis=1)
    # Save as .npy (you can also save as PNG if needed)
    save_path = os.path.join(output_dir, f"interpolated_step_{idx:04d}.npy")
    np.save(save_path, phase_slm)
    plt.imshow(phase_slm, cmap='gray')
    plt.colorbar()
    plt.title(f"Interpolated Step {idx}")
    #lt.show()

