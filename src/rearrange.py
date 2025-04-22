import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage.feature import peak_local_max
import matplotlib.patches as patches
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from time import sleep
import time

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


def tweez_fourier_scaled(target_im_ideal):

    wavelength = 813e-9  
    focal_length = 300e-3 
    pixel_pitch_slm = 8e-6  
    pixel_pitch_ccd = 3.45e-6  

    delta_x = (wavelength * focal_length) / (SIZE_X * pixel_pitch_slm)  
    delta_y = (wavelength * focal_length) / (SIZE_Y * pixel_pitch_slm)  

    scale_x = delta_x / pixel_pitch_ccd
    scale_y = delta_y / pixel_pitch_ccd

    # print(f"Fourier Scaling Δx = {delta_x:.3e} m, Δy = {delta_y:.3e} m")
    # print(f"Scaling in CCD pixels: scale_x = {scale_x:.2f}, scale_y = {scale_y:.2f}")

    # # === Map Tweezer Positions from CCD to SLM ===
    tweezer_positions_ccd = np.argwhere(target_im_ideal == 1)  
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


def match_tweezers(detected_positions, target_positions):
    """
    Assign detected tweezers to target positions optimally.

    Args:
    - detected_positions: (Nx2) array of detected tweezer positions in Fourier space.
    - target_positions: (Nx2) array of target tweezer positions.

    Returns:
    - assignments: List of (initial_position, final_position) pairs.
    """
    cost_matrix = cdist(detected_positions, target_positions, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return list(zip(detected_positions[row_ind], target_positions[col_ind]))

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def move_tweezers(phase, assignments, slm_size=(1200, 1920)):
    """
    Moves tweezers to new positions while keeping amplitude and phase the same.

    Args:
    - phase: The original phase pattern (2D numpy array).
    - assignments: List of (initial_position, final_position) mappings.
    - slm_size: (height, width) of SLM.

    Returns:
    - new_phase_pattern: The updated phase pattern for the SLM.
    """

    # Compute Fourier Transform of the Phase Pattern
    u_fft = fftshift(fft2(np.exp(1j * phase)))  # Get Fourier space representation

    # Extract Amplitude and Phase of Tweezers
    amplitude = np.abs(u_fft)  
    phase_fft = np.angle(u_fft)  

    # Create a new blank Fourier space
    u_fft_new = np.zeros_like(u_fft, dtype=np.complex_)

    # Move each tweezer by copying phase & amplitude to the new position
    for (start, end) in assignments:
        amp = amplitude[start[0], start[1]]  # Extract amplitude
        ph = phase_fft[start[0], start[1]]  # Extract phase

        u_fft_new[end[0], end[1]] = amp * np.exp(1j * ph)  # Assign to new position

    amplitude = np.abs(u_fft_new)  
    phase_fft = np.angle(u_fft_new)  

    # Transform back to real space
    new_phase = np.angle(ifft2(ifftshift(u_fft_new)))

    return new_phase

# Load the .npy file
file_path = "final_phase_pattern.npy"  # Adjust if needed
phase_pattern = np.load(r"C:\Users\Yb\SLM\SLM\data\images\final_phase_pattern.npy")

SIZE_Y,SIZE_X=phase_pattern.shape


# Apply FFT to get the tweezer pattern
u_fft = fft2(np.exp(1j * phase_pattern))  # Correctly apply FFT to the phase pattern
u_fft = fftshift(u_fft)

# Extract Amplitude and Phase
amplitude = np.abs(u_fft)  # Intensity of tweezers in Fourier space
phase = np.angle(u_fft)  # Phase information of tweezers


# Generate target positions in a circle
circle_positions = generate_circle_positions(25, (1200, 1920))  # Get (y, x) coordinates

# Create a binary mask for the target image
target_im_ideal = np.zeros((1200, 1920), dtype=np.uint8)  # Create blank image
target_im_ideal[circle_positions[:, 0], circle_positions[:, 1]] = 1  # Set tweezers to 1
target_positions = tweez_fourier_scaled(target_im_ideal)  # Convert to Fourier space
coordinates_new = peak_local_max(
        target_positions,
        min_distance=8,
        num_peaks=25
    )

coordinates_old = peak_local_max(
    np.abs(u_fft),
    min_distance=10,
    num_peaks=25
)

# Extract phase and amplitude at detected tweezer positions
tweezer_amplitudes = amplitude[coordinates_old[:, 0], coordinates_old[:, 1]]  # Get amplitudes at tweezer locations
tweezer_phases = phase[coordinates_old[:, 0], coordinates_old[:, 1]]  # Get phases at tweezer locations


# Assuming `coordinates_old` contains the detected tweezer positions
# and `coordinates_new` contains the target positions
assignments = match_tweezers(coordinates_old, coordinates_new)

new_phase = move_tweezers(phase_pattern, assignments)


new_phase = np.round((new_phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)


import os
import numpy as np
from ctypes import *
from scipy import misc
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt
from pypylon import pylon
#import cv2

def open_camera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.ExposureTime.SetValue(50)  
    camera.PixelFormat.SetValue("Mono8") 
    return camera


################################ MAKE SURE THE WINDOW SHOWS UP IN THE WRITE PLACE FOR THE DPI SETTINGS#############
# Query DPI Awareness (Windows 10 and 8)
import ctypes
awareness = ctypes.c_int()
errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
print(awareness.value)

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


###test image
#imgs_arrays = generate_bmp_images_central_on()
#yimgs_arrays = np.load(r"C:\Users\Yb\Documents\slm_calibration\Superpixel_images_8_12\superpixel_image_qmax_8_12_label_36.npy")


#spots_correction_blaze = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\24_01_test\spots_with_correction.npy")
#spots_correction_blaze_tot =  np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\23_01_test\spots_with_correction_2pi_tot.npy")
#spots_blaze = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\16_01_test\spots_with_grating.npy")
#grating_with_correction = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\16_01_test\grating_with_correction.npy")
#blazed_grating = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\blazed_diagonal_8pps.npy")
#spots_blaze_wfc  = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\15_01_test\spots_with_grating_wfc.npy") 
four_spots = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\four_spots_15it.npy")
#correction_mask = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\NEW_correction_map.npy")
#spots_with_mask = four_spots - correction_mask 
#lens_pattern_with_correction = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\21_01_test\lens_pattern_with_correction.npy")
#lens_pattern_blaze = np.load(r"C:\Users\thoma\Documents\Edu\PhD\Simulations\SLM\Calibration\21_01_test\lens_pattern_blaze.npy")

#camera = open_camera()
#phase_correction  = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_img_fit_mod.npy")
#spots = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_holo8bit.npy")
#blazed_grating = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\blazed_grating.npy")
#image_send = phase_correction + spots + blazed_grating
# get to be displayed on SLM
#ImageOne = imgs_arrays[...,31].flatten()
#print(ImageOne.shape)

blazed_grating = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\blazed_grating.npy")
#phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_correction_19_02.npy")
#phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_correction_20_02_relative.npy")
phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_0_2pi_25_02.npy")
five_grid = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_5x5_grid_1920x1200_100pixels.npy")


image_to_send = blazed_grating + phase_correction + five_grid# + new_phase

plt.figure(figsize=(10, 5))
plt.imshow(image_to_send, cmap='twilight', extent=[0, phase.shape[1], 0, phase.shape[0]])
plt.colorbar(label='Phase (radians)')
plt.title("Image to send SLM")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")

image_to_send = blazed_grating + phase_correction + new_phase
plt.figure(figsize=(10, 5))
plt.imshow(image_to_send, cmap='twilight', extent=[0, phase.shape[1], 0, phase.shape[0]])
plt.colorbar(label='Phase (radians)')
plt.title("Image to send")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
#plt.show()

#plt.show()
#image_to_send = blazed_grating + + phase_correction


# h, w = image_to_send.shape
# new_h, new_w = 1024, 1024

# start_row = (h - new_h) // 2  # 88
# start_col = (w - new_w) // 2  # 448
# 
# # Create a new array of zeros with the same shape as the original image
# masked_img = np.zeros_like(image_to_send)

# # Copy the central (1024, 1024) region into the new array
# masked_img[start_row:start_row+new_h, start_col:start_col+new_w] = image_to_send[start_row:start_row+new_h, start_col:start_col+new_w]

#print(masked_img.dtype)

# plt.figure()
# plt.imshow(masked_img)
# plt.show()
#phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_img_fit_mod.npy")
#phase_pattern_first = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\1x1_grid_1920x1200_phase.npy")
#phase_pattern_first = phase_pattern_first[:,::-1]
#first_img_to_SLM  = blazed_grating #+ phase_correction + phase_pattern_first

# imgs_arrays = np.load(r"C:\Users\Yb\Documents\slm_calibration\superpixels_images_60_60\superpixel_image_qmax_1_12_label_38.npy")
# for i_counter in range(len(imgs_arrays[0,0,:])):
#    slm_lib.Write_image(imgs_arrays[...,i_counter].flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);
#    sleep(25) 

#slm_lib.Write_image(imgs_arrays[...,88].flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);
#sleep(1000)


#slm_lib.Write_image(image_to_send.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);
#sleep(1000)


# Define the two patterns
image1 = blazed_grating + phase_correction + new_phase
image2 = blazed_grating + phase_correction + five_grid  # Alternative pattern

# Infinite loop for oscillation
while True:
    slm_lib.Write_image(image1.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image)
    time.sleep(50)  # Adjust delay if needed

    slm_lib.Write_image(image2.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image)
    #time.sleep(0.5)  # Adjust delay if needed

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK();
print ("Blink SDK was successfully deleted");

