import os
import numpy as np
from ctypes import *
from scipy import misc
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt
from pypylon import pylon
#import cv2



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

four_spots = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\four_spots_15it.npy")
blazed_grating = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\blazed_grating.npy")
phase_correction = np.load(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\phase_map_0_2pi_25_02.npy")
five_grid = np.load(r"C:\Users\Yb\SLM\SLM\data\images\final_phase_pattern_triangle.npy")
five_grid=np.flip(five_grid, axis=1)


# Directory containing the phase patterns
folder_path = r"C:\Users\Yb\SLM\SLM\data\images"

# Loop over steps 1 to 10
for i in range(0, 11):
    file_name = f"tweezer_step_{i}_phase.npy"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        print(f"Loading: {file_path}")
        image_to_send = np.load(file_path)

        # If needed, flip the image to match SLM coordinate system
        image_to_send = np.flip(image_to_send, axis=1) + blazed_grating + phase_correction

        # Send image to SLM
        slm_lib.Write_image(image_to_send.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image)
        sleep(1) # 1 second pause between each frame
    else:
        print(f"File not found: {file_path}")


# Always call Delete_SDK before exiting
slm_lib.Delete_SDK();
print ("Blink SDK was successfully deleted");

