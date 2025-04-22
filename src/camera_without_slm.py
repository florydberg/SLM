import os
import numpy as np
from time import time, strftime
from PIL import Image
from pypylon import pylon
import csv

# ------------------ Paths ------------------
output_dir = r"C:\Users\Yb\SLM\SLM\data\captured_frames"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "frame_timestamps.csv")

# ------------------ Camera Setup ------------------
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(100)
camera.PixelFormat.SetValue("Mono8")
camera.Width.Value = 900
camera.Height.Value = 900
camera.OffsetX.Value = 416
camera.OffsetY.Value = 151
camera.AcquisitionMode.SetValue("Continuous")
camera.StartGrabbing()

print(f"Camera theoretical frame rate: {camera.ResultingFrameRate.Value:.2f} FPS\n")

# ------------------ Capture Loop ------------------
frame_counter = 0
dur_sec = 10.0
start_time = time()
timestamps = []
last_hw_timestamp = None

with open(log_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame_index", "hw_timestamp_ns", "sw_timestamp_s"])

    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            hw_timestamp = grab_result.TimeStamp  # in nanoseconds
            sw_timestamp = time()

            frame = grab_result.Array
            np.save(os.path.join(output_dir, f"frame_{frame_counter:04d}.npy"), frame)
            #Image.fromarray(frame).save(os.path.join(output_dir, f"frame_{frame_counter:04d}.png"))

            writer.writerow([frame_counter, hw_timestamp, sw_timestamp])
            timestamps.append((frame_counter, hw_timestamp, sw_timestamp))

            # Compute interval from last frame (based on hardware timestamp)
            if last_hw_timestamp is not None:
                dt_us = (hw_timestamp - last_hw_timestamp) / 1000  # in microseconds
            else:
                dt_us = 0

            last_hw_timestamp = hw_timestamp

            # Print formatted timing info
            print(f"[CAMERA] Frame {frame_counter:03d} | HW: {hw_timestamp/1e6:.3f} ms | "
                  f"SW: {sw_timestamp:.6f} s | Δt: {dt_us:.1f} µs")

            frame_counter += 1

        grab_result.Release()

        if sw_timestamp - start_time > dur_sec:
            break

# ------------------ Cleanup ------------------
camera.StopGrabbing()
camera.Close()

print(f"\n[CAMERA] Done. {frame_counter} frames captured.")
print(f"[CSV] Timestamps saved to: {log_file}")
