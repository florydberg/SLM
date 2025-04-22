import os
import numpy as np
from ctypes import *
from time import time
from pypylon import pylon  # Only needed if you also use it in your SLM code, else remove it.
from time import sleep 
import time
# ------------------ Paths and Initialization ------------------
slm_folder = r"C:\Users\Yb\SLM\SLM\data\images"
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")
is_eight_bit_image = c_uint(1)
slm_lib.Create_SDK() 
refresh_period = .1  # seconds
blazed_grating = np.load(r"C:\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\blazed_grating.npy")
phase_correction = np.load(r"C:\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\phase_map_0_2pi_25_02.npy")

# ------------------ Pattern Update Loop ------------------
pattern_filenames = sorted([f for f in os.listdir(slm_folder) if f.endswith('.npy') and 'tweezer_step' in f])
total_patterns = len(pattern_filenames)

for pattern_index, fname in enumerate(pattern_filenames):
    pattern_path = os.path.join(slm_folder, fname)
    pattern = np.load(pattern_path)
    # Process the pattern (flip, add gratings/corrections, etc.)
    pattern = np.flip(pattern, axis=1) + blazed_grating + phase_correction
    slm_lib.Write_image(pattern.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image)
    print(f"[SLM] Pattern {pattern_index+1}/{total_patterns} sent: {fname}")
    time.sleep(refresh_period)


slm_lib.Delete_SDK()
print("All SLM patterns sent.")
