import shutil

# Define the path to the output directory and the zip file name
output_dir = '/kaggle/working/60fps_converted-videos'
zip_file = '/kaggle/working/60fps_converted-videos.zip'

# Create a zip archive of the output folder
shutil.make_archive(zip_file.replace('.zip', ''), 'zip', output_dir)

# Verify the creation of the zip file
os.listdir('/kaggle/working/')
