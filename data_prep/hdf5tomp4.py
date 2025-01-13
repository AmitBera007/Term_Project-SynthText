import h5py

hdf5_file = 'input_file.hdf5'

with h5py.File(hdf5_file, 'r') as f:
    # List all datasets inside the file
    print(f.keys())


import cv2
import numpy as np

# Path to the HDF5 file
hdf5_file = 'input_file.hdf5'

# Open the HDF5 file
with h5py.File(hdf5_file, 'r') as f:
    # Access the 'video' dataset
    video_data = f['video'][:]

# Assuming the video data is a sequence of frames in numpy format
# Convert each frame to a video file (MP4 format)
output_file = 'output_video.mp4'
height, width = video_data.shape[1], video_data.shape[2]  # Assuming shape is (num_frames, height, width, channels)
fps = 60  # Adjust this based on your source data

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Iterate through frames and write them to the output video
for frame in video_data:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
    out.write(frame_bgr)

# Release the video writer
out.release()

print(f"Video saved as {output_file}")
