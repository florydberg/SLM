import os
import numpy as np
from ctypes import *
from time import time, sleep
from PIL import Image
from pypylon import pylon
import threading

# ------------------ Paths ------------------
slm_folder = r"C:\Users\Yb\SLM\SLM\data\images"
output_dir = r"C:\Users\Yb\SLM\SLM\data\captured_frames"
os.makedirs(output_dir, exist_ok=True)

# ------------------ Load DLLs ------------------
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")
RGB = c_uint(0)
is_eight_bit_image = c_uint(1)
slm_lib.Create_SDK()

# ------------------ Load phase components ------------------
blazed_grating = np.load(r"C:\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\blazed_grating.npy")
phase_correction = np.load(r"C:\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK\\phase_map_0_2pi_25_02.npy")

# ------------------ Camera Setup ------------------
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(500)  # 0.5 ms
camera.PixelFormat.SetValue("Mono8")
camera.Width.Value = 900
camera.Height.Value = 900
camera.OffsetX.Value = 564
camera.OffsetY.Value = 194
camera.StartGrabbing()

print(f"Camera resulting frame rate: {camera.ResultingFrameRate.Value:.2f} FPS")

# ------------------ Pattern Preparation ------------------
pattern_filenames = sorted([f for f in os.listdir(slm_folder) if f.endswith('.npy') and 'tweezer_step' in f])
total_patterns = len(pattern_filenames)
slm_done = False


# ------------------ Camera thread function ------------------
def camera_runner():
    frame_counter = 0
    start_time = time()
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            frame = grab_result.Array
            Image.fromarray(frame).save(os.path.join(output_dir, f"frame_{frame_counter:04d}.png"))
            np.save(os.path.join(output_dir, f"frame_{frame_counter:04d}.npy"), frame)
            print(f"[CAMERA] Captured frame {frame_counter}")
            frame_counter += 1
        grab_result.Release()

        # Stop after SLM is done + a few buffer frames
        if slm_done and frame_counter > total_patterns + 5:
            break

    total_time = time() - start_time
    print(f"[CAMERA] Done. Captured {frame_counter} frames in {total_time:.2f} s → Approx. {frame_counter / total_time:.2f} FPS.")


# ------------------ Start camera in background ------------------
camera_thread = threading.Thread(target=camera_runner)
camera_thread.start()

# ------------------ SLM update loop in main thread ------------------
print("Starting SLM updates in main thread...")

for pattern_index, fname in enumerate(pattern_filenames):
    pattern_path = os.path.join(slm_folder, fname)
    pattern = np.load(pattern_path)
    pattern = np.flip(pattern, axis=1) + blazed_grating + phase_correction
    slm_lib.Write_image(pattern.flatten().ctypes.data_as(POINTER(c_ubyte)), is_eight_bit_image)
    print(f"[SLM] Pattern {pattern_index+1}/{total_patterns} sent: {fname}")

slm_done = True
print("[SLM] All patterns sent.")

# ------------------ Wait for camera to finish ------------------
camera_thread.join()

# ------------------ Cleanup ------------------
camera.StopGrabbing()
camera.Close()
slm_lib.Delete_SDK()
print("Finished all threads and cleaned up.")
