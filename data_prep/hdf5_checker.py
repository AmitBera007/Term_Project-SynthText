import h5py

def explore_hdf5_file(file_path):
    """
    Function to explore the structure and contents of an HDF5 file.
    
    Parameters:
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'r') as hdf:
        # Print the structure of the HDF5 file
        def print_structure(name, obj):
            print(f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset shape: {obj.shape}")
                print(f"  Dataset dtype: {obj.dtype}")
                # Optionally print some data (e.g., first 5 elements)
                # print(f"  Data (first 5 elements): {obj[0:5]}")
        
        print(f"Exploring file: {file_path}")
        hdf.visititems(print_structure)

# Provide the path to your HDF5 file
hdf5_file_path = "hdf5_folder/1001_IEO_ANG_HI.hdf5"
explore_hdf5_file(hdf5_file_path)


# Extract video
import cv2
import os

def extract_video_from_hdf5(hdf5_file, output_video_path, fps=25):
    # Open the HDF5 file
    with h5py.File(hdf5_file, 'r') as f:
        # Access the video dataset
        video_data = f['video'][:]
        num_frames, height, width, _ = video_data.shape

        # Initialize the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Write each frame to the video
        for i, frame in enumerate(video_data):
            # Convert the frame from RGB to BGR (as OpenCV uses BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)  # Write the frame to the video file

        # Release the VideoWriter object
        out.release()

        print(f"Saved video to {output_video_path}")

# Usage
extract_video_from_hdf5('hdf5_folder/1001_IEO_ANG_HI.hdf5', 'output_video.mp4', fps=25)
