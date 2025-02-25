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

def intensity_std(u_int, coordinates):
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
    # Extract intensity values at detected peak locations
    peak_intensities = u_int[coordinates[:, 0], coordinates[:, 1]]

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


def sort_peaks_x(coordinates, x_tolerance=10):
    """
    Sorts detected peaks:
    - Groups peaks with similar X-coordinates
    - Sorts each group by Y-coordinate

    Args:
    - coordinates (ndarray): Nx2 array of (Y, X) peak coordinates.
    - x_tolerance (int): Max difference in X-values to consider them part of the same group.

    Returns:
    - sorted_coordinates (ndarray): Peaks ordered in a stable manner.
    """
    # Step 1: Sort primarily by X (column-wise)
    coordinates = coordinates[np.argsort(coordinates[:, 1])]

    # Step 2: Group similar X-values and sort by Y within each group
    sorted_list = []
    start = 0
    for i in range(1, len(coordinates)):
        if abs(coordinates[i, 1] - coordinates[start, 1]) > x_tolerance:
            # Sort the current X-group by Y and add to the final list
            sorted_list.extend(sorted(coordinates[start:i], key=lambda p: p[0]))  # Sort by Y
            start = i  # Start new group

    # Sort last group
    sorted_list.extend(sorted(coordinates[start:], key=lambda p: p[0]))

    return np.array(sorted_list)

import numpy as np

def sort_peaks_by_y(coordinates, y_tolerance=10):
    """
    Sorts detected peaks:
    - Groups peaks with similar Y-coordinates
    - Sorts each group by X-coordinate

    Args:
    - coordinates (ndarray): Nx2 array of (Y, X) peak coordinates.
    - y_tolerance (int): Max difference in Y-values to consider them part of the same group.

    Returns:
    - sorted_coordinates (ndarray): Peaks ordered in a stable manner.
    """
    # Step 1: Sort primarily by Y (row-wise)
    coordinates = coordinates[np.argsort(coordinates[:, 0])]

    # Step 2: Group similar Y-values and sort by X within each group
    sorted_list = []
    start = 0
    for i in range(1, len(coordinates)):
        if abs(coordinates[i, 0] - coordinates[start, 0]) > y_tolerance:
            # Sort the current Y-group by X and add to the final list
            sorted_list.extend(sorted(coordinates[start:i], key=lambda p: p[1]))  # Sort by X
            start = i  # Start new group

    # Sort last group
    sorted_list.extend(sorted(coordinates[start:], key=lambda p: p[1]))

    return np.array(sorted_list)

def weights_(w, target, w_prev, std_int):

    coordinates = peak_local_max(
        std_int,
        min_distance=70,
        num_peaks=25
    )

    coordinates = sort_peaks_by_y(coordinates)

    w_prev_values = w_prev[target==1]
    std_int_values = std_int[coordinates[:, 0], coordinates[:, 1]]

    avg_w_prev = np.mean(w_prev_values)   
    avg_std_int = np.mean(std_int_values) 

    w[target == 1] = np.sqrt((w_prev_values / avg_w_prev) / (std_int_values / avg_std_int))

    #w_updated = w.copy()  
    #w_updated[coordinates[:, 0], coordinates[:, 1]] = new_weights
    #print(w_updated.shape)


    return w, coordinates


def weights(w, target, w_prev, std_int): 
    # Detect peaks in the CCD intensity image
    coordinates = peak_local_max(
        std_int,
        min_distance=70,
        num_peaks=25
    )



    #coordinates = coordinates[np.lexsort((coordinates[:,0], coordinates[:,1]))] # 0 = y coordinates and 1 is x coordinates
    coordinates = sort_peaks_by_y(coordinates)
    # Update weights using detected tweezer positions
    #print("\n🔹 Raw Detected Coordinates (Y, X):")
    #for i, (y, x) in enumerate(coordinates):
    #    print(f"  Peak {i+1}: (Y={y}, X={x})")

    w[target == 1] = np.sqrt(target[target == 1] / std_int[coordinates[:, 0], coordinates[:, 1]]) * w_prev[target == 1]
    
    # # # Plot the detected tweezer positions with numbering
    # plt.figure(figsize=(10, 6))
    # plt.imshow(target, cmap='gray')  # Display target intensity image
    # plt.scatter(coordinates[:, 1], coordinates[:, 0], c='red', marker='x', s=100, label="Detected Tweezers")

    # # Add numbering to each detected tweezer
    # for i, (x, y) in enumerate(coordinates):
    #     plt.text(y, x, str(i+1), color="yellow", fontsize=12, ha='center', va='center', fontweight='bold')

    # # Labels & formatting
    # plt.title("Detected Tweezer Positions with Numbering")
    # plt.xlabel("X pixels")
    # plt.ylabel("Y pixels")
    # plt.legend()
    # plt.colorbar(label="Intensity")
    # plt.show()
    # pause(20)
    return w, coordinates

def weights_a(w, target, w_prev, std_int): 
    # Detect peaks in the CCD intensity image
    coordinates = peak_local_max(
        std_int,
        min_distance=70,
        num_peaks=25
    )

    coordinates = sort_peaks_by_y(coordinates)
    
    # Extract measured intensities at detected tweezer positions
    A_measured = std_int[coordinates[:, 0], coordinates[:, 1]]

    # Extract previous weights at tweezer positions
    T_prev = target[target == 1]  # T_prev is the target values at tweezer positions
    avg_T_prev = np.mean(T_prev)  # Average of previous target intensities

    # Compute the new weights
    w[target == 1] = (T_prev / avg_T_prev) * (A_measured / np.mean(A_measured)) * w_prev[target == 1]

    return w

#exponential update
def weights_exp(w, target, w_prev, std_int, gain, mask):
    """
    Update weights using an exponential update rule.
    """
    update_factor = np.exp(gain * (target- std_int))
    # Only update where mask is True.
    w = w_prev
    w =w_prev * update_factor
    #w[mask] = w_prev[mask] * update_factor[mask]
    
    return w

def basler():
        #print("Camera model:", camera.GetDeviceInfo().GetModelName())
    camera.StartGrabbingMax(1)  
    grab_result = camera.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)
    
    if grab_result.GrabSucceeded():
        frame = grab_result.Array
        std_int = frame # Keep it normalized to [0, 1]
        std_int = np.flip(std_int, axis=1)  # Mirror along the x-axis
 

    return std_int

# --- Main Code ---

# Number of iterations
n_rep = 100

target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\adjusted_5x5_grid_100pixels.npy")
target_im_ideal = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\5x5_grid_1920x1200_100pixels.npy")
target_im=norm(target_im)   # Image in intensity units [0,1]
target_im_ideal = norm(target_im_ideal)
SIZE_Y,SIZE_X=target_im.shape

# Plot the target image
fig, axs = plt.subplots(2, 2)
im0 = axs[0, 0].imshow(target_im_ideal)
plt.colorbar(im0, ax=axs[0, 0])
axs[0, 0].set_title('Target image')




# Create the original mask (points where target_im is 1)
mask_points = (target_im ==1)
# Dilate the mask to consider a 3x3 region around each True pixel.
mask = binary_dilation(mask_points, structure=np.ones((20, 20)))



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
camera.ExposureTime.SetValue(200)
camera.PixelFormat.SetValue("Mono8") 

blazed_grating = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\blazed_grating.npy")
phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_img_fit_mod.npy")
phase_pattern_first = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_5x5_grid_1920x1200_100pixels.npy")

first_img_to_SLM  = blazed_grating + phase_correction + phase_pattern_first

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


im1 = axs[0, 1].imshow(w)
axs[0, 1].set_title('w')

im2 = axs[1, 0].imshow(phase_pattern_first + blazed_grating + phase_correction, vmin=-pi, vmax=pi)
axs[1, 0].set_title('Phase')

im3 = axs[1, 1].imshow(np.zeros((1200,1920)), vmin=0, vmax=1)
axs[1, 1].set_title('Reconstructed image')

cb1 = fig.colorbar(im1, ax=axs[0, 1])
cb2 = fig.colorbar(im2, ax=axs[1, 0])
cb3 = fig.colorbar(im3, ax=axs[1, 1])

#print(f"Type of phase: {phase_pattern_first[544]}")
# Iterate with a progress bar


# Initialize the figure and axis for the error plot
plt.ion()  # Interactive mode ON
fig, ax = plt.subplots(figsize=(6, 4))
error_line, = ax.plot([], [], 'r-')  # Initialize an empty plot
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.set_title("Error Convergence")
ax.set_ylim(0, 1)  # Set a fixed y-limit (adjust if needed)
ax.set_xlim(0, n_rep)  # Adjust to total iterations

errors = []  # Store error values

#phase = np.zeros((SIZE_X, SIZE_Y), dtype=np.uint8)

# Convert to SLM range [0, 255]
#phase_slm = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)  
init_ampl=np.sqrt(target_im) # Amplitude of the tweezer spots i.e. A = sqrt(intensity)
#w_prev = target_im # as done by mezzanti

# Initial random phase in the range [-pi, pi]
phase = 2 * np.pi * np.random.rand(SIZE_Y, SIZE_X) - np.pi
#phase = np.zeros((SIZE_Y, SIZE_X))

for rep in tqdm(range(n_rep), desc="Iterations", unit="it"):

    # phase_2pi = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8) #convert phase in [0, 2pi) range and then convert to 8-bit
    # phase_2pi = np.round((phase) * 255 / (2 * np.pi)).astype(np.uint8) # convert to 8-bit

    #if rep==0:
    pattern_to_slm = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8) + blazed_grating + phase_correction # this part is the phase pattern to send to the slm
    #else:
    #    pattern_to_slm = phase_2pi

    slm_lib.Write_image(pattern_to_slm.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);
    pause(0.2) #wait 100 ms for lcs to settle

    
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

    if rep>i:

        ###### Weights ########
        w,coordinates=weights(w,target_im,w_prev,std_int)
        w=norm(w)
        w_prev=w.copy()
        #u=join_phase_ampl(phase,w*target_im)
        u=join_phase_ampl(phase,w)

        ############## Plotting ##################
        # Find the error between ideal target image and basler image
        #error_value = epsilon(std_int, target_im_ideal)
        error_value = intensity_std(std_int, coordinates)
        errors.append(error_value)

        # Update the plot dynamically
        error_line.set_xdata(np.arange(len(errors)))
        error_line.set_ydata(errors)
        ax.set_xlim(0, len(errors) + 1)  # Extend x-axis as needed
        plt.draw()
        plt.pause(0.1)  # Small pause to update the plot

        # Plots to show phase and 
        im2.set_data(phase)
        im3.set_data(std_int)
        plt.tight_layout()
        plt.pause(0.1)
        ############## Plotting ##################
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

    #################### This part goes back to phases being from -pi to pi #################################

    #phase = np.round((phase_rad + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)

    #phase = phase[:,::-1] 
    #Final_ampl_phase = phase.copy()  # Final discretized phase (if needed)
    
    
    
plt.figure()
plt.plot(np.arange(n_rep), errors, "-o")
plt.yscale('log')  # Set y-axis to log scale
plt.ylim(min(errors) * 0.1, max(errors) * 10)  # Adjust limits for better visualization
plt.xlabel("Iteration")
plt.ylabel("Epsilon (Error)")
plt.title("Epsilon Convergence (Log Scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid for better readability
plt.savefig(r"C:\Users\Yb\SLM\SLM\data\images\gsw_logerrors.png")  
plt.show()    

plt.figure()
plt.plot(np.arange(n_rep), errors, "-o")
#plt.yscale('log')
plt.ylim(1e-2,1)
plt.savefig(r"C:\Users\Yb\SLM\SLM\data\images\gsw_errors.png")  
#print(errors)
#grab_result.Release()
#import numpy as np

# np.save("errors.npy", errors)
# output_path_smooth = "errors.bmp"
# Image.fromarray((errors)).save(errors)


np.save(r"C:\Users\Yb\SLM\SLM\data\images\final_ccd_image", std_int)



camera.StopGrabbing()
camera.Close()
#np.save("Final_ampl_phase.npy", Final_ampl_phase)
# Plot convergence, reconstructed image, and error difference

plt.show()