import numpy as np
import matplotlib.pyplot as plt
from scipy import fft as sfft
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter
from tqdm import tqdm  # progress bar
from PIL import Image
import random
import imageio
import sys
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import png as png
import os
from ctypes import *
from time import sleep
from pypylon import pylon
import cv2
from skimage.feature import peak_local_max
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment
import glob

def epsilon(u_int, target_im_ideal, neighborhood=80):
    """
    Compute the relative intensity error for a 5x5 tweezer array.
    
    Args:
    - u_int: 2D array of measured intensities
    - target_im: 2D array indicating tweezer positions
    - neighborhood: size of the local region around each tweezer to consider for peak detection

    Returns:
    - error: (I_max - I_min) / (I_max + I_min)
    """
    tweezer_positions = np.argwhere(target_im_ideal == 1)  # Get x, y coordinates of tweezers
    peak_intensities = []

    for x, y in tweezer_positions:
        # Define neighborhood limits
        x_min = max(0, x - neighborhood // 2)
        x_max = min(u_int.shape[0], x + neighborhood // 2 + 1)
        y_min = max(0, y - neighborhood // 2)
        y_max = min(u_int.shape[1], y + neighborhood // 2 + 1)

        # Extract local region and find max intensity
        local_region = u_int[x_min:x_max, y_min:y_max]
        peak_intensity = np.max(local_region)
        #peak_intensity = np.sum(np.sum(local_region))
        peak_intensities.append(peak_intensity)
    #print("\n Peak Intensities ", peak_intensities)
    # Compute error metric

    mean_intensity = np.mean(peak_intensities)
    std_intensity = np.std(peak_intensities)

    # Avoid division by zero
    relative_std = std_intensity / mean_intensity if mean_intensity > 0 else 1.0

    return relative_std

def intensity_std(u_int, coordinates_ccd):
    """
    Compute the standard deviation of tweezer intensities using peak detection.
    
    Args:
    - u_int: 2D array of measured intensities.
    - min_distance: Minimum distance between detected peaks.
    - num_peaks: Expected number of tweezers (peaks) to detect.

    Returns:
    - relative_std: Standard deviation of intensities among tweezers normalized by mean intensity.
    - peak_intensities: List of peak intensities at each tweezer location.
    - coordinates: List of detected tweezer coordinates.
    """

    # coordinates_ccd = peak_local_max(
    #     std_int,
    #     min_distance=3,
    #     num_peaks=4
    # )

    # Extract intensity values at detected peak locations
    peak_intensities = u_int[coordinates_ccd[:, 0], coordinates_ccd[:, 1]]

    # Compute normalized standard deviation
    mean_intensity = np.mean(peak_intensities)
    std_intensity = np.std(peak_intensities)
    #print(peak_intensities)

    # Avoid division by zero
    relative_std = std_intensity / mean_intensity if mean_intensity > 0 else 1.0

    return relative_std

def join_phase_ampl(phase, ampl):
    """Combine phase and amplitude (vectorized)."""
    return ampl * np.exp(1j * phase)

# def Beam_shape(sizex, sizey, sigma, mu):
#     """Generate a Gaussian beam shape."""
#     x, y = np.meshgrid(np.linspace(-1, 1, sizex), np.linspace(-1, 1, sizey))
#     d = np.sqrt(x * x + y * y)
#     return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

def norm(matrix):
    """Normalize matrix values to the range [0, 1]."""
    matrix_min = matrix.min()
    matrix_max = matrix.max()

    return (matrix - matrix_min) / (matrix_max - matrix_min)


def match_detected_to_target(detected, target):
    # detected and target are (N, 2) arrays
    cost_matrix = np.linalg.norm(detected[:, np.newaxis, :] - target[np.newaxis, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #row_ind → indices from the detected array
    #col_ind → indices from the target array
    return row_ind, col_ind  # Detected index i maps to target index col_ind[i]

def weights(w, target, w_prev, std_int, coordinates_ccd_first, min_distance, num_peaks): 
    # Detect peaks in the CCD intensity image

    
    # coordinates_ccd_first = peak_local_max(
    #     std_int,
    #     min_distance=3,
    #     num_peaks=4
    # )
    coordinates_t = np.argwhere(target > 0)  # don't use peak_local_max for target
    # Step 1: Create empty image of same shape
    image_shape = std_int.shape
    ccd_mask = np.zeros(image_shape, dtype=std_int.dtype)  # same type as std_int

    # Step 2: Fill in only the detected peaks with their intensity
    for y, x in coordinates_ccd_first:
        ccd_mask[y, x] = std_int[y, x]

    std_int = tweez_fourier_scaled_std(ccd_mask)
    coordinates_ccd = peak_local_max(
        std_int,
        min_distance=min_distance,
        num_peaks=num_peaks    )

    # plt.figure(figsize=(8, 6))
    # plt.imshow(std_int, cmap='gray')
    # plt.title("Fourier Scaled Intensity (std_int)")
    # plt.xlabel("X Pixels")
    # plt.ylabel("Y Pixels")
    # plt.colorbar(label='Intensity')
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()
    # plt.pause(30)



    row_ind, col_ind = match_detected_to_target(coordinates_ccd, coordinates_t)
    # print(row_ind)
    # print(col_ind)
    # # ccd_coords = coordinates_ccd[row_ind]

    # plt.figure(figsize=(10, 8))
    # plt.imshow(std_int, cmap='gray', alpha=0.5)

    # # Plot CCD-detected peaks
    # plt.scatter(coordinates_ccd[:, 1], coordinates_ccd[:, 0], c='red', marker='x', label='Detected (CCD)')

    # # Plot Target peaks
    # plt.scatter(coordinates_t[:, 1], coordinates_t[:, 0], c='cyan', marker='o', facecolors='none', label='Target (Ideal)')

    # # Draw matching lines
    # for i, j in zip(row_ind, col_ind):
    #     p1 = coordinates_ccd[i]  # Detected point (y, x)
    #     p2 = coordinates_t[j]    # Target point (y, x)
    #     plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'yellow', linestyle='--', linewidth=1)

    # # Optionally: Add numbering
    # for i, (y, x) in enumerate(coordinates_ccd):
    #     plt.text(x, y, str(i), color='red', fontsize=9, ha='center', va='center')

    # for j, (y, x) in enumerate(coordinates_t):
    #     plt.text(x, y, str(j), color='cyan', fontsize=9, ha='center', va='center')

    # plt.title("Matched Detected (CCD) and Target Tweezer Positions")
    # plt.xlabel("X Pixels")
    # plt.ylabel("Y Pixels")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    # pause(50)

    # # # # Plot the detected tweezer positions with numbering
    # plt.figure(figsize=(10, 6))
    # plt.imshow(target_im_ideal, cmap='gray')  # Display target intensity image
    # plt.scatter(matched_coordinates[:, 1], matched_coordinates[:, 0], c='red', marker='x', s=100, label="Detected Tweezers")

    # # Add numbering to each detected tweezer
    # for i, (x, y) in enumerate(matched_coordinates):
    #     plt.text(y, x, str(i+1), color="yellow", fontsize=12, ha='center', va='center', fontweight='bold')

    # # Labels & formatting
    # plt.title("Detected Tweezer Positions with Numbering")
    # plt.xlabel("X pixels")
    # plt.ylabel("Y pixels")
    # plt.legend()
    # plt.colorbar(label="Intensity")
    # plt.show()
    # #pause(100)
    #w[target == 1] = np.sqrt(target[target == 1] / std_int[matched_coordinates[:, 0], matched_coordinates[:, 1]]) * w_prev[target == 1]
    # Get matched detected (CCD) and target (ideal) coordinates
    ccd_coords = coordinates_ccd[row_ind]   # shape (25, 2)
    tgt_coords = coordinates_t[col_ind]     # shape (25, 2)

    # Separate y, x for indexing
    ccd_y, ccd_x = ccd_coords[:, 0], ccd_coords[:, 1]
    tgt_y, tgt_x = tgt_coords[:, 0], tgt_coords[:, 1]

    # Gather intensities and ideal target values
    intensities = std_int[ccd_y, ccd_x]
    ideal_values = target[tgt_y, tgt_x]
    previous_weights = w_prev[tgt_y, tgt_x]


    # Compute new weights
    updated_weights = np.sqrt(ideal_values / intensities) * previous_weights

    # Assign updated weights
    w[tgt_y, tgt_x] = updated_weights

    return w



def basler():
        #print("Camera model:", camera.GetDeviceInfo().GetModelName())
    camera.StartGrabbingMax(1)  
    grab_result = camera.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)
    
    if grab_result.GrabSucceeded():
        frame = grab_result.Array
        std_int = frame # Keep it normalized to [0, 1]
        #std_int = np.flip(std_int, axis=1)  # Mirror along the x-axis
 

    return std_int

def tweez_fourier_scaled_std(target_im_ideal):
    wavelength = 813e-9  
    focal_length = 300e-3 
    pixel_pitch_slm = 8e-6  
    pixel_pitch_ccd = 3.45e-6  
    SIZE_Y, SIZE_X = target_im_ideal.shape

    delta_x = (wavelength * focal_length) / (SIZE_X * pixel_pitch_slm)  
    delta_y = (wavelength * focal_length) / (SIZE_Y * pixel_pitch_slm)  

    scale_x = delta_x / pixel_pitch_ccd
    scale_y = delta_y / pixel_pitch_ccd

    # Get all non-zero (or high intensity) positions
    tweezer_positions_ccd = np.argwhere(target_im_ideal > 0)

    # Convert CCD pixels to real-world distances
    tweezer_positions_ccd_x = (tweezer_positions_ccd[:, 1] - SIZE_X / 2) * pixel_pitch_ccd
    tweezer_positions_ccd_y = (tweezer_positions_ccd[:, 0] - SIZE_Y / 2) * pixel_pitch_ccd

    # Apply Fourier scaling
    tweezer_positions_slm_x = tweezer_positions_ccd_x / delta_x
    tweezer_positions_slm_y = tweezer_positions_ccd_y / delta_y

    # Shift back to pixel grid
    tweezer_positions_slm_x = np.round(tweezer_positions_slm_x + SIZE_X / 2).astype(int)
    tweezer_positions_slm_y = np.round(tweezer_positions_slm_y + SIZE_Y / 2).astype(int)

    # Initialize output image
    target_im = np.zeros_like(target_im_ideal)

    # Copy intensity values from original to scaled locations
    for orig, new_x, new_y in zip(tweezer_positions_ccd, tweezer_positions_slm_x, tweezer_positions_slm_y):
        orig_y, orig_x = orig
        if 0 <= new_y < SIZE_Y and 0 <= new_x < SIZE_X:
            target_im[new_y, new_x] = target_im_ideal[orig_y, orig_x]

    return target_im


def tweez_fourier_scaled(target_im_ideal):

    wavelength = 813e-9  
    focal_length = 300e-3 
    pixel_pitch_slm = 8e-6  
    pixel_pitch_ccd = 3.45e-6  
    SIZE_Y,SIZE_X=target_im_ideal.shape

    delta_x = (wavelength * focal_length) / (SIZE_X * pixel_pitch_slm)  
    delta_y = (wavelength * focal_length) / (SIZE_Y * pixel_pitch_slm)  

    scale_x = delta_x / pixel_pitch_ccd
    scale_y = delta_y / pixel_pitch_ccd

    # print(f"Fourier Scaling Δx = {delta_x:.3e} m, Δy = {delta_y:.3e} m")
    # print(f"Scaling in CCD pixels: scale_x = {scale_x:.2f}, scale_y = {scale_y:.2f}")

    # # === Map Tweezer Positions from CCD to SLM ===
    tweezer_positions_ccd = np.argwhere(target_im_ideal == 1)  
    #tweezer_positions_ccd = np.argwhere(target_im_ideal > 0.5)  # or >= 0.99
    # Convert CCD pixels to real-world distances

    tweezer_positions_ccd_x = (tweezer_positions_ccd[:, 1] - SIZE_X / 2) * pixel_pitch_ccd
    tweezer_positions_ccd_y = (tweezer_positions_ccd[:, 0] - SIZE_Y / 2) * pixel_pitch_ccd

    # Apply Fourier Scaling to map to SLM space
    tweezer_positions_slm_x = tweezer_positions_ccd_x / delta_x
    tweezer_positions_slm_y = tweezer_positions_ccd_y / delta_y

    # Shift back to CCD coordinate space
    tweezer_positions_slm_x = np.round(tweezer_positions_slm_x + SIZE_X / 2).astype(int)
    tweezer_positions_slm_y = np.round(tweezer_positions_slm_y + SIZE_Y / 2).astype(int)

    target_im = np.zeros((SIZE_Y, SIZE_X), dtype=np.uint8)
    target_im[tweezer_positions_slm_y, tweezer_positions_slm_x] = 255  
    # # Plot Phase Pattern
    # plt.figure(figsize=(8, 6))
    # plt.imshow(target_im)
    # plt.colorbar(label="8-bit values")
    # plt.title("PScaled Tweezer Positions")
    # plt.xlabel("SLM X")
    # plt.ylabel("SLM Y")
    # plt.show()
    return target_im


def generate_circle_positions(num_tweezers, img_shape, radius=300):
    """
    Generate target tweezer positions arranged in a circle.
    
    Args:
    - num_tweezers: Number of tweezers to position.
    - img_shape: Tuple (height, width) of the Fourier space.
    - radius: Radius of the circle in pixels.

    Returns:
    - target_positions: Array of (y, x) coordinates for tweezers in a circle.
    """
    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2  # Center of the image
    angles = np.linspace(0, 2 * np.pi, num_tweezers, endpoint=False)  # Equally spaced angles
    target_positions = np.array([
        (center_y + radius * np.sin(theta), center_x + radius * np.cos(theta)) for theta in angles
    ])
    
    return np.round(target_positions).astype(int)



# --- Main Code ---

#target_im_ideal = np.load(r"C:\Users\Yb\SLM\SLM\data\target_images\random_spaced_tweezers.npy")
target_im_ideal = np.load(r"C:\Users\Yb\SLM\SLM\data\target_images\2x2tweezers.npy")

# Number of iterations
n_rep = 30
#target_im = np.load(r"C:\Users\Yb\SLM\SLM\data\target_images\square_1920x1200.npy")
#target_im_ideal = np.load(r"C:\Users\Yb\SLM\SLM\data\target_images\square_1920x1200.npy")
#target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\adjusted_5x5_grid_100pixels.npy")
#target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\adjusted_5x5_grid_100pixels.npy")
#target_im_ideal = np.load(r"C:\Users\Yb\SLM\SLM\notebooks\Tweezer_step_images\tweezer_step_0008.npy")

#target_im = neww() 
target_im = tweez_fourier_scaled(target_im_ideal)

target_im=norm(target_im)   # Image in intensity units [0,1]
target_im_ideal = norm(target_im_ideal)
SIZE_Y,SIZE_X=target_im.shape
#target_im = tweez_fourier_scaled(target_im_ideal)


# plt.figure(figsize=(8, 6))
# plt.imshow(target_im, cmap='gray', origin='upper')  # Flip origin if needed
# plt.title("Target Image After Fourier Scaling")
# plt.xlabel("X pixels")
# plt.ylabel("Y pixels")
# plt.colorbar(label="Intensity")
# plt.tight_layout()
# plt.show()
# #pause(10)




# ******
# Plot the target image
fig, axs = plt.subplots(2, 2)
im0 = axs[0, 0].imshow(target_im_ideal)
plt.colorbar(im0, ax=axs[0, 0])
axs[0, 0].set_title('Target image')



# Initialize weights and beam shape
#w = np.ones((SIZE_Y, SIZE_X))
w = target_im.copy()
#PS_shape = Beam_shape(SIZE_X, SIZE_Y, sigma=255, mu=0)
#PS_shape = Beam_shape(SIZE_X, SIZE_Y, sigma=300, mu=0)
PS_shape = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\incident_amplitude.npy")
PS_shape = norm(PS_shape)

w_prev = target_im.copy()  # initial previous weights
#w_prev = np.zeros_like(target_im)
errors = []
#errors = np.array()
u = np.zeros((SIZE_Y, SIZE_X), dtype=complex)
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(900)
camera.PixelFormat.SetValue("Mono8") 

blazed_grating = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\blazed_grating.npy")
phase_correction = np.load(r"C:\Users\Yb\Documents\slm_calibration\psi_images_avg_10_04_04_25\phase_map_0_2pi_09_04.npy")

#phase_pattern_first = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_5x5_grid_1920x1200_100pixels.npy")

first_img_to_SLM  = blazed_grating + phase_correction #+ phase_pattern_first

# Query DPI Awareness (Windows 10 and 8)
import ctypes
awareness = ctypes.c_int()
errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))

# Set DPI Awareness  (Windows 10 and 8)
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
# the argument is the awareness level, which can be 0, 1 or 2:
# for 1-to-1 pixel control I seem to need it to be non-zero (I'm using level 2)

# Set DPI Awareness  (Windows 7 and Vista)
success = ctypes.windll.user32.SetProcessDPIAware()
# behaviour on later OSes is undefined, although when I run it on my Windows 10 machine, it seems to work with effects identical to SetProcessDpiAwareness(1)
#######################################################################################################################


# Load the DLL
# Blink_C_wrapper.dll, HdmiDisplay.dll, ImageGen.dll, freeglut.dll and glew64.dll
# should all be located in the same directory as the program referencing the
# library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

# Open the image generation library --> use for generating images through Meadwlark's custom functions
#cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\ImageGen")
#image_lib = CDLL("ImageGen")

# indicate that our images are greyscale 8 bit
RGB = c_uint(0);
is_eight_bit_image = c_uint(1);

# Call the constructor
init_result = slm_lib.Create_SDK();

# ******

im1 = axs[0, 1].imshow(w)
axs[0, 1].set_title('w')

#im2 = axs[1, 0].imshow(phase_pattern_first + blazed_grating + phase_correction, vmin=-pi, vmax=pi)
im2 = axs[1, 0].imshow(blazed_grating + phase_correction, vmin=-pi, vmax=pi)
axs[1, 0].set_title('Phase')

im3 = axs[1, 1].imshow(np.zeros((1200,1920)), vmin=0, vmax=1)
axs[1, 1].set_title('Reconstructed image')

cb1 = fig.colorbar(im1, ax=axs[0, 1])
cb2 = fig.colorbar(im2, ax=axs[1, 0])
cb3 = fig.colorbar(im3, ax=axs[1, 1])

# #print(f"Type of phase: {phase_pattern_first[544]}")
# # Iterate with a progress bar


# Initialize the figure and axis for the error plot
plt.ion()  # Interactive mode ON
fig, ax = plt.subplots(figsize=(6, 4))
error_line, = ax.plot([], [], 'r-')  # Initialize an empty plot
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.set_title("Error Convergence")
ax.set_ylim(0, 1)  # Set a fixed y-limit (adjust if needed)
ax.set_xlim(0, n_rep)  # Adjust to total iterations
#******

errors = []  # Store error values

#phase = np.zeros((SIZE_X, SIZE_Y), dtype=np.uint8)

# Convert to SLM range [0, 255]
#phase_slm = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)  
init_ampl=np.sqrt(tweez_fourier_scaled(target_im_ideal)) # Amplitude of the tweezer spots i.e. A = sqrt(intensity)
#w_prev = target_im # as done by mezzanti

# Initial random phase in the range [-pi, pi]
phase = 2 * np.pi * np.random.rand(SIZE_Y, SIZE_X) - np.pi
#phase = np.zeros((SIZE_Y, SIZE_X))
min_distance = 4
num_peaks = 25


phase_history = np.zeros((n_rep, 25))  # max_reps = total iterations



for rep in tqdm(range(n_rep), desc="Iterations", unit="it"):

    # phase_2pi = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8) #convert phase in [0, 2pi) range and then convert to 8-bit
    # phase_2pi = np.round((phase) * 255 / (2 * np.pi)).astype(np.uint8) # convert to 8-bit

    #if rep==0:

    pattern_to_slm = np.flip(np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8), axis=1) + blazed_grating + phase_correction # this part is the phase pattern to send to the slm
    #else:
    #    pattern_to_slm = phase_2pi

    slm_lib.Write_image(pattern_to_slm.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);
    pause(0.02) #wait 100 ms for lcs to settle

    
    # Convert back to phase in [-π, π] range for the FFT
    #################### This part goes back to phases being from -pi to pi #################################
    #phase= (phase_slm / 255) * 2 * np.pi - np.pi # i see a problem with thiiiisssss dsjölkdsajlksjklvndslnvkdsanlkndsnfsd
    # Apply FFT and update phase
    u = join_phase_ampl(phase, PS_shape)
    u = sfft.fft2(u)
    u = sfft.fftshift(u)


    # Normalized intensity coming from actual images from the Basler
    std_int = basler()
    std_int = std_int/255 # if not try std_int / 255!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ******
    coordinates = peak_local_max(
         std_int,
         min_distance=min_distance,
         num_peaks=num_peaks
     )
    # Plots to show phase and 
    im2.set_data(phase)
    im3.set_data(std_int)
    plt.tight_layout()
    plt.pause(0.01)

    phase=np.angle(u) # This is from -pi to pi


    i = 3
    if rep<i:
        u = join_phase_ampl(phase, init_ampl)


    # else:
    #     ###### Weights ########
    #     w=weights(target_im,w_prev,std_int)
    #     w=norm(w)
    #     w_prev=w
    #     u=join_phase_ampl(phase,w)
    #     ###### Weights ########

    # if rep>i :
    #     ###### Weights ########

    #     w=weights(w,target_im,w_prev,std_int, coordinates, min_distance, num_peaks)
    #     #w, coordinates = weigths_box(w,target_im,w_prev,std_int)
    #     # random_weight_noise = 0.008 * np.random.randn(*w.shape)
    #     # w += random_weight_noise
    #     # w = np.clip(w, 0, None)  # Ensure weights stay positive
    #     w=norm(w)
    #     coordinates_last = peak_local_max(
    #      w,
    #      min_distance=min_distance,
    #      num_peaks=num_peaks
    #     )
    #     w_prev=w.copy()
    #     #u=join_phase_ampl(phase,w*target_im)

    #     u=join_phase_ampl(phase,w)

    # if rep>last rep:
    #     ###### Weights ########

    #     w=weights(w,target_im,w_prev,std_int, coordinates, min_distance, num_peaks)

    #     #w, coordinates = weigths_box(w,target_im,w_prev,std_int)
    #     w=norm(w)
    #     w_prev=w.copy()
    #     #u=join_phase_ampl(phase,w*target_im)
    #     u=join_phase_ampl(phase,w)

        # u = sfft.ifftshift(u)
        # u = sfft.ifft2(u)
        # phase = np.angle(u)
        # u = sfft.fft2(u)
        # u = sfft.fftshift(u)

    if rep > i& i<24:
        w=weights(w,target_im,w_prev,std_int, coordinates, min_distance, num_peaks)
        #w, coordinates = weigths_box(w,target_im,w_prev,std_int)
        # random_weight_noise = 0.008 * np.random.randn(*w.shape)
        # w += random_weight_noise
        # w = np.clip(w, 0, None)  # Ensure weights stay positive
        w=norm(w)
        coordinates_last = peak_local_max(
        w,
        min_distance=min_distance,
        num_peaks=num_peaks
        )
        w_prev=w.copy()
        #u=join_phase_ampl(phase,w*target_im)
 
        # Start with a zero phase array
        masked_phase = np.zeros_like(phase)

        # Unpack coordinates
        ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]

        # Keep the phase values only at the detected peaks
        masked_phase[ys, xs] = phase[ys, xs]
        #asked_phase[ys, xs] = np.pi / 2
        current_phases = masked_phase[ys, xs]
        phase_history[rep, :] = current_phases  # each row is a repetition
        plt.figure(figsize=(8, 6))
        plt.imshow(masked_phase)  # Flip origin if needed
        plt.title("Target Image After Fourier Scaling")
        plt.xlabel("X pixels")
        plt.ylabel("Y pixels")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        plt.show()
        pause(100)
        amplitude_peaks = np.zeros_like(np.abs(u))
        # Extract the peak positions
        ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]
        amplitude_peaks[ys, xs] = np.abs(u)[ys, xs]
        #u=join_phase_ampl(phase,w)
        random_phases = np.zeros_like(w)
        random_values = 2 * np.pi * np.random.rand(len(ys)) - np.pi
        random_phases[ys, xs] = random_values
        field_with_random_phase = w * np.exp(1j * random_phases)
        u=join_phase_ampl(masked_phase,w) 
  
       # u=join_phase_ampl(phase,amplitude_peaks)
        # random_phases = np.zeros_like(amplitude_peaks)
        # random_values = 2 * np.pi * np.random.rand(len(ys)) - np.pi
        # random_phases[ys, xs] = random_values
        # # plt.figure(figsize=(8, 6))
        # # plt.imshow(random_phases)  # Flip origin if needed
        # # plt.title("Target Image After Fourier Scaling")
        # # plt.xlabel("X pixels")
        # # plt.ylabel("Y pixels")
        # # plt.colorbar(label="Intensity")
        # # plt.tight_layout()

        # plt.figure(figsize=(8, 6))
        # plt.imshow(amplitude_peaks)  # Flip origin if needed
        # plt.title("Target Image After Fourier Scaling")
        # plt.xlabel("X pixels")
        # plt.ylabel("Y pixels")
        # plt.colorbar(label="Intensity")
        # plt.tight_layout()
        # plt.show()
        # pause(100)
        # # Step 2 — Generate random phases between [-π, π] at the same peak positions

        # # Step 3 — Construct complex field: amplitude × exp(i × phase)
        # field_with_random_phase = amplitude_peaks * np.exp(1j * random_phases)
        # u = field_with_random_phase
        ########### Plotting ##################
        #Find the error between ideal target image and basler image
        #error_value = epsilon(std_int, target_im_ideal)

    # if rep ==25:

    #     w=weights(w,target_im,w_prev,std_int, coordinates, min_distance, num_peaks)
    #     #w, coordinates = weigths_box(w,target_im,w_prev,std_int)
    #     # random_weight_noise = 0.008 * np.random.randn(*w.shape)
    #     # w += random_weight_noise
    #     # w = np.clip(w, 0, None)  # Ensure weights stay positive
    #     w=norm(w)
    #     coordinates_last = peak_local_max(
    #     w,
    #     min_distance=min_distance,
    #     num_peaks=num_peaks
    #     )
    #     w_prev=w.copy()
    #     #u=join_phase_ampl(phase,w*target_im)


    #     # Start with a zero phase array
    #     masked_phase = np.zeros_like(phase)

    #     # Unpack coordinates
    #     ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]

    #     # Keep the phase values only at the detected peak
    #     random_vals = 2 * np.pi * np.random.rand(len(ys)) - np.pi
    #     masked_phase[ys, xs] = random_vals
    #     #asked_phase[ys, xs] = phase[ys, xs]
    #     #asked_phase[ys, xs] = np.pi / 2
    #     current_phases = masked_phase[ys, xs]
    #     phase_history[rep, :] = current_phases  # each row is a repetition
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(masked_phase)  # Flip origin if needed
    #     plt.title("Target Image After Fourier Scaling")
    #     plt.xlabel("X pixels")
    #     plt.ylabel("Y pixels")
    #     plt.colorbar(label="Intensity")
    #     plt.tight_layout()
    #     plt.show()
    #     pause(100)
    #     amplitude_peaks = np.zeros_like(np.abs(u))
    #     # Extract the peak positions
    #     ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]
    #     amplitude_peaks[ys, xs] = np.abs(u)[ys, xs]
    #     #u=join_phase_ampl(phase,w)
    #     random_phases = np.zeros_like(w)
    #     random_values = 2 * np.pi * np.random.rand(len(ys)) - np.pi
    #     random_phases[ys, xs] = random_values
    #     field_with_random_phase = w * np.exp(1j * random_phases)
    #     u=join_phase_ampl(masked_phase,w) 
    # if rep > 25:
    #     w=weights(w,target_im,w_prev,std_int, coordinates, min_distance, num_peaks)
    #     #w, coordinates = weigths_box(w,target_im,w_prev,std_int)
    #     # random_weight_noise = 0.008 * np.random.randn(*w.shape)
    #     # w += random_weight_noise
    #     # w = np.clip(w, 0, None)  # Ensure weights stay positive
    #     w=norm(w)
    #     coordinates_last = peak_local_max(
    #     w,
    #     min_distance=min_distance,
    #     num_peaks=num_peaks
    #     )
    #     w_prev=w.copy()
    #     #u=join_phase_ampl(phase,w*target_im)


    #     # Start with a zero phase array
    #     masked_phase = np.zeros_like(phase)

    #     # Unpack coordinates
    #     ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]

    #     # Keep the phase values only at the detected peaks
    #     masked_phase[ys, xs] = phase[ys, xs]
    #     #asked_phase[ys, xs] = np.pi / 2
    #     current_phases = masked_phase[ys, xs]
    #     phase_history[rep, :] = current_phases  # each row is a repetition
    #     # plt.figure(figsize=(8, 6))
    #     # plt.imshow(masked_phase)  # Flip origin if needed
    #     # plt.title("Target Image After Fourier Scaling")
    #     # plt.xlabel("X pixels")
    #     # plt.ylabel("Y pixels")
    #     # plt.colorbar(label="Intensity")
    #     # plt.tight_layout()
    #     # plt.show()
    #     # pause(100)
    #     amplitude_peaks = np.zeros_like(np.abs(u))
    #     # Extract the peak positions
    #     ys, xs = coordinates_last[:, 0], coordinates_last[:, 1]
    #     amplitude_peaks[ys, xs] = np.abs(u)[ys, xs]
    #     #u=join_phase_ampl(phase,w)
    #     random_phases = np.zeros_like(w)
    #     random_values = 2 * np.pi * np.random.rand(len(ys)) - np.pi
    #     random_phases[ys, xs] = random_values
    #     field_with_random_phase = w * np.exp(1j * random_phases)
    #     u=join_phase_ampl(masked_phase,w) 
  

    error_value = intensity_std(std_int, coordinates)
    errors.append(error_value)
    # ******

        # Update the plot dynamically
    error_line.set_xdata(np.arange(len(errors)))
    error_line.set_ydata(errors)
    ax.set_xlim(0, len(errors) + 1)  # Extend x-axis as needed
    plt.draw()
    plt.pause(0.01)  # Small pause to update the plot
        # ******

    
    ############# Plotting ##################
    print(errors[-1])



        # plt.figure(figsize=(8, 6))
        # plt.imshow(w, cmap='hot')
        # plt.colorbar(label="Weight Values")
        # plt.title(f"Weights After {rep} Iterations")
        # plt.show()
        #pause(2)



    u = sfft.ifftshift(u)
    u = sfft.ifft2(u)
    


    # The phase that we need to imprint by the SLM is:
    phase = np.angle(u) # This is still -pi to pi!!






    # if rep == 19:
    #     # Apply FFT to get the tweezer pattern
    #     u_fft = fft2(phase)
    #     u_fft = fftshift(u_fft)

    #     # Detect peaks in the FFT magnitude spectrum
    #     coordinates = peak_local_max(
    #         np.abs(u_fft),
    #         min_distance=10,
    #         num_peaks=25
    #         )
    #     circle_positions = generate_circle_positions(25, (1200, 1920))  # Get (y, x) coordinates

    #     # Create a binary mask for the target image
    #     target_im_ideal = np.zeros((1200, 1920), dtype=np.uint8)  # Create blank image
    #     target_im_ideal[circle_positions[:, 0], circle_positions[:, 1]] = 1  # Set tweezers to 1
    #     target_positions = tweez_fourier_scaled(target_im_ideal)  # Convert to Fourier space


    #################### This part goes back to phases being from -pi to pi #################################

    #phase = np.round((phase_rad + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)

    #phase = phase[:,::-1] 
    #Final_ampl_phase = phase.copy()  # Final discretized phase (if needed)

        






unwrapped_history = np.unwrap(phase_history, axis=0)  # unwrap along time/repetition axis

plt.figure(figsize=(12, 6))
for idx in range(len(xs)):
    plt.plot(unwrapped_history[:, idx], label=f'Pixel {idx}')
plt.xlabel('Repetition')
plt.ylabel('Unwrapped Phase Value (radians)')
plt.title('Unwrapped Phase Evolution at Each Tweezer Pixel')
plt.legend(loc='upper right', fontsize='small', ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.pause(100)
# plt.figure()
# plt.plot(np.arange(n_rep), errors, "-o")
# plt.yscale('log')  # Set y-axis to log scale
# plt.ylim(min(errors) * 0.1, max(errors) * 10)  # Adjust limits for better visualization
# plt.xlabel("Iteration")
# plt.ylabel("Epsilon (Error)")
# plt.title("Epsilon Convergence (Log Scale)")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid for better readability
# plt.savefig(r"C:\Users\Yb\SLM\SLM\data\images\gsw_logerrors.png")  
# plt.show()    

# plt.figure()
# plt.plot(np.arange(n_rep), errors, "-o")
# #plt.yscale('log')
# plt.ylim(1e-2,1)
# plt.savefig(r"C:\Users\Yb\SLM\SLM\data\images\gsw_errors.png")  
#print(errors)
#grab_result.Release()
#import numpy as np

# Save log-scale plot of the convergence errors

# === Set the base name once ===
#basename = "tweezer_step_0008"
output_folder = r"C:\Users\Yb\SLM\SLM\data\images"
# ******

# # === Plot log-scale error convergence ===
# plt.figure(figsize=(8, 5))
# plt.plot(np.arange(len(errors)), errors, "-o")
# plt.yscale('log')
# plt.xlabel("Iteration")
# plt.ylabel("Relative Std (log scale)")
# plt.title("Error Convergence Over Iterations")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.tight_layout()

log_error_path = os.path.join(output_folder, f"gsw_errors_log_{basename}.png")
plt.savefig(log_error_path)
print(f"Saved log error plot to: {log_error_path}")
# ******

# # === Plot final CCD image ===
# plt.figure(figsize=(10, 8))
# plt.imshow(std_int, cmap='gray')
# plt.title("Final CCD Image")
# plt.xlabel("X Pixels")
# plt.ylabel("Y Pixels")
# plt.colorbar(label="Intensity")
# plt.tight_layout()

final_ccd_path = os.path.join(output_folder, f"final_ccd_image_{basename}.png")
plt.savefig(final_ccd_path)
print(f"Saved final CCD image plot to: {final_ccd_path}")
#plt.show()

# === Save final phase as .npy or .png ===
phase_uint8 = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)
npy_save_path = os.path.join(output_folder, f"{basename}_slm.npy")
np.save(npy_save_path, phase_uint8)
print(f"Saved final phase to: {npy_save_path}")
# === Save final phase as .npy or .png ===
npy_save_path = os.path.join(output_folder, f"{basename}.npy")
np.save(npy_save_path, phase)
print(f"Saved final phase to: {npy_save_path}")

# === Close camera ===
camera.StopGrabbing()
camera.Close()