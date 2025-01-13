import os
import shutil
import argparse
import random

def divide_hdf5_files(folder_path, train_hdf_folder_path, val_hdf_folder_path, val_ratio=0.25):
    # Create the directories if they don't exist
    os.makedirs(train_hdf_folder_path, exist_ok=True)
    os.makedirs(val_hdf_folder_path, exist_ok=True)

    # Get a list of all HDF5 files in the folder
    hdf_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.hdf5')]

    # Shuffle the files randomly
    random.shuffle(hdf_files)

    # Calculate the split index
    split_index = int(len(hdf_files) * (1 - val_ratio))

    # Split the files into training and validation sets
    train_files = hdf_files[:split_index]
    val_files = hdf_files[split_index:]

    # Copy the training files to the training folder
    for hdf_file in train_files:
        src_path = os.path.join(folder_path, hdf_file)
        dest_path = os.path.join(train_hdf_folder_path, hdf_file)
        shutil.copy(src_path, dest_path)
        print(f"File {hdf_file} copied to {train_hdf_folder_path}.")

    # Copy the validation files to the validation folder
    for hdf_file in val_files:
        src_path = os.path.join(folder_path, hdf_file)
        dest_path = os.path.join(val_hdf_folder_path, hdf_file)
        shutil.copy(src_path, dest_path)
        print(f"File {hdf_file} copied to {val_hdf_folder_path}.")

    print("\nAll HDF5 files have been divided into training and validation sets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide HDF5 files into train and validation sets.")
    parser.add_argument("-w", "--folder_path", type=str, required=True, help="Path to the folder containing HDF5 files")
    parser.add_argument("-t", "--train_hdf_folder_path", type=str, required=True, help="Path to the folder for training HDF5 files")
    parser.add_argument("-v", "--val_hdf_folder_path", type=str, required=True, help="Path to the folder for validation HDF5 files")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Ratio of files to be used for validation (default: 0.25 for 25%)")
    args = parser.parse_args()

    divide_hdf5_files(args.folder_path, args.train_hdf_folder_path, args.val_hdf_folder_path, args.val_ratio)

