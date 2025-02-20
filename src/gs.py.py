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

# def epsilon(u_int, mask):
#      """Compute the relative intensity error at the tweezers positions."""
#      vals = u_int[mask]
#      print("vals:, vals.shape:",vals, vals.shape)
#      max_val = np.max(vals)
#      min_val = np.min(vals)
#      print(f"\nEpsilon max value:  {max_val}\n")
#      print(f"Epsilon min value:  {min_val}")
#      return (max_val - min_val) / (max_val + min_val)

# def epsilon(u_int, target_im):
#     max = np.max(u_int[target_im==1]) #Max value of the obtained intensity at the tweezers position
#     min = np.min(u_int[target_im==1]) #Min value of the obtained intensity at the tweezers position
#     #print("min: {}".format(min))
#     #plt.figure()
#     #plt.imshow(u_int[target_im==1])
#     # Get the indices where target_im == 1
  

#     error =  (max-min)/(max+min)
#     #print("Error :", error)
#     return error

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
    I_max = np.max(peak_intensities)
    I_min = np.min(peak_intensities)
    error = (I_max - I_min) / (I_max + I_min)
    if np.isnan(error):
        error = 1

    return error





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

# def weights(w, target_im, w_prev, std_int, mask):
#     """Update weights only at the tweezers positions."""
#     w[mask] = np.sqrt(target_im[mask] / std_int[mask]) * w_prev[mask]
#     return w

# def weights(w, target_im, w_prev, std_int, mask, mask_points):
    
#     #update_factor = np.sqrt(np.mean(np.mean(target_im[mask].reshape(len(target_im[mask_points]), 3, 3), axis=1),axis=1) * np.mean(np.mean(std_int[mask].reshape(len(std_int[mask_points]), 3, 3), axis=1),axis=1))
#     #w[mask] = w_prev[mask] * update_factor.repeat(9)
#     update_factors  = np.sqrt(np.mean(np.mean(std_int[mask].reshape(len(std_int[mask_points]), 3, 3), axis=1),axis=1))
#     w[mask_points] = np.sum(update_factors)/update_factors * w_prev[mask_points]
#     #w[mask_points] = w_prev[mask_points] * update_factors
#     return w


def weights(w,target, w_prev, std_int, coord_prev): # This weight function works only for discrete tweezers
    
    # # Mask the std_int image so that
    ymin = 400
    ymax = 800
    xmin = 750
    xmax = 1200
    #roi = std_int[ymin:ymax, xmin:xmax]
    image_shape = std_int.shape

    mask = np.zeros(image_shape, dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    masked_im = std_int * mask
    # 2) Run peak_local_max on the cropped array

    coordinates = peak_local_max(
        masked_im,
        min_distance=70,
        #threshold_abs=np.max(std_int)/10
    )
    
    # 3) Offset the coordinates back into the full array
    #coords_full = coords_in_roi + [ymin, xmin]

    # Now coords_full has the correct (row, col) in the original image
    #coordinates = coords_full
    # roi_mask = np.zeros_like(std_int, dtype=bool)
    # roi_mask[400:800, 700:1200] = True
    # coordinates = peak_local_max(
    # std_int, 
    # min_distance=80,                    # require peaks to be at least min_distance pixels apart
    # threshold_abs=np.max(std_int)/5,
    # mask = roi_mask    # only accept peaks above threshold intensity
    # )
    # #print(target[target==1].shape)

    # === Define SLM & CCD Parameters ===
    slm_pixel_pitch = 8e-6  # SLM pixel pitch (8 µm per pixel)
    ccd_pixel_size = 3.45e-6  # CCD camera pixel size (3.45 µm per pixel)
    image_height, image_width = target.shape  # CCD image resolution

    # Laser & Fourier Optics Setup
    wavelength = 813e-9  # Laser wavelength (meters)
    focal_length = 300e-3  # Lens focal length (meters)

    # === Compute Fourier Transform Scaling Factors ===
    delta_x = (wavelength * focal_length) / (image_width * slm_pixel_pitch)  # Fourier transform resolution in meters
    delta_y = (wavelength * focal_length) / (image_height * slm_pixel_pitch)

    # Convert from meters to CCD pixels
    scale_x = delta_x / ccd_pixel_size
    scale_y = delta_y / ccd_pixel_size                   

    # Converting to integers (camera needs pixels)
    slm_x = np.round(coordinates[:, 1] / scale_x).astype(int)
    slm_y = np.round(coordinates[:, 0] / scale_y).astype(int)
    
    prev_x = np.round(coord_prev[:, 1] / scale_x).astype(int)
    prev_y = np.round(coord_prev[:, 0] / scale_y).astype(int)
    
    # Weights using slm coordinates, not ccd coordinates
    # std_int is the image from the ccd camera, target is the target image
    w[slm_y, slm_x] = np.sqrt(target[target == 1]/std_int[coordinates[:, 0], coordinates[:, 1]])* w_prev[prev_y, prev_x]
    return w, coordinates
 
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
        
        # Normalize the image to [0, 1]
        #normalized_frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) )
        #normalized_frame =  (frame) / (np.max(frame))
        #normalized_frame_8bit = (frame).astype(np.uint8)
        
        # Save as .tiff with Pillow
        #output_filename = f"output_image_{rep}.bmp"
        #Image.fromarray(normalized_frame_8bit).save(output_filename)
        #print(f"Image saved as {output_filename}")
        #frame  = frame/255
        # Update std_int for the next iteration
        std_int = frame # Keep it normalized to [0, 1]
        #std_int = normalized_frame
        #cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)  # Make the window resizable
        #cv2.resizeWindow("Camera Output", 800, 600)  # Set the desired window size
        #cv2.imshow("Camera Output", normalized_frame)
        #cv2.waitKey(0)


    #cv2.destroyAllWindows()

    return std_int

# --- Main Code ---

# Number of iterations
n_rep = 150
# if n_rep <= 0 or n_rep > 150:
#     raise ValueError("Wrong number of iterations")


#target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\tweezers_positions_updated.npy")
#target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\image_plane_spots.npy")
#target_im = np.load(r"C:\Users\Yb\Downloads\tweezer_position_cam.npy")
#target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\5x5_grid_corrected_for_FT.npy")
#target_im = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\singletweezer.npy")
#target_im_ideal = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\singletweezer.npy")

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
w = np.ones((SIZE_Y, SIZE_X))
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
camera.ExposureTime.SetValue(1000)
camera.PixelFormat.SetValue("Mono8") 

blazed_grating = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\blazed_grating.npy")
phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_img_fit_mod.npy")
#phase_pattern_first = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_pattern_to_try.npy")
phase_pattern_first = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_5x5_grid_1920x1200_100pixels.npy")
#phase_pattern_first = np.load(r"C:\Users\Yb\Downloads\phase_map_5x5_grid_1920x1200_100pixels.npy")
#phase_pattern_first = phase_pattern_first[:,::-1]
first_img_to_SLM  = blazed_grating + phase_correction + phase_pattern_first

# Initial random phase in the range [-pi, pi]
#phase = 2 * np.pi * np.random.rand(SIZE_X, SIZE_Y) - np.pi


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

# Initialize phase from -pi to pi
phase = np.random.rand(SIZE_Y, SIZE_X) * 2 * np.pi - np.pi 

# Convert to SLM range [0, 255]
#phase_slm = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)  

init_ampl=np.sqrt(target_im) # Amplitude of the tweezer spots i.e. A = sqrt(intensity)




for rep in tqdm(range(n_rep), desc="Iterations", unit="it"):

    phase_2pi = np.round((phase + np.pi) * 255 / (2 * np.pi)).astype(np.uint8) #convert phase in [0, 2pi) range and then convert to 8-bit

    #if rep==0:
    pattern_to_slm = phase_2pi + blazed_grating + phase_correction # this part is the phase pattern to send to the slm
    #else:
    #    pattern_to_slm = phase_2pi

    slm_lib.Write_image(pattern_to_slm.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image);
    sleep(0.2) #wait 100 ms for lcs to settle
    
    # Convert back to phase in [-π, π] range for the FFT

    #################### This part goes back to phases being from -pi to pi #################################
    phase = (phase_2pi / 255) * 2 * np.pi - np.pi  

    # Apply FFT and update phase
    u = join_phase_ampl(phase, PS_shape)
    u = sfft.ifftshift(u)
    u = sfft.ifft2(u)

    # Normalized intensity coming from actual images from the Basler
    std_int = basler()
    std_int = std_int/255 # if not try std_int / 255!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    ############## Plotting ##################
    # Find the error between ideal target image and basler image
    error_value = epsilon(std_int, target_im_ideal)
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
    if error_value<0.2:
        plt.pause(20)


    phase=np.angle(u) # This is from -pi to pi
    #alpha = 0.5  # Blending factor (tune between 0.2 - 0.7)
    #phase = alpha * np.angle(u) + (1 - alpha) * phase

    u = join_phase_ampl(phase, init_ampl)
    u = sfft.ifftshift(u)
    u = sfft.ifft2(u)

    # The phase that we need to imprint by the SLM is:
    phase = np.angle(u) # This is still -pi to pi!!

    #################### This part goes back to phases being from -pi to pi #################################

    #phase = np.round((phase_rad + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)

    #phase = phase[:,::-1] 
    #Final_ampl_phase = phase.copy()  # Final discretized phase (if needed)
    print(errors)
    
    

plt.figure()
plt.plot(np.arange(n_rep), errors, "-o")
#plt.yscale('log')
plt.ylim(1e-2,1)
plt.savefig("errors.png")  
#print(errors)
#grab_result.Release()
#import numpy as np

# np.save("errors.npy", errors)
# output_path_smooth = "errors.bmp"
# Image.fromarray((errors)).save(errors)


np.save("image_on_camera_5_5.npy", std_int)



camera.StopGrabbing()
camera.Close()
#np.save("Final_ampl_phase.npy", Final_ampl_phase)
# Plot convergence, reconstructed image, and error difference

plt.show()