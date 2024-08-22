
import geopandas as gpd
from os.path import join, basename, exists
import os

# #FOR GETTING ALL THE PATCH NAMES OF ALL S2 TILES

mosaic_folder = '../data_preprocessing/cropped_mosaic'

# Get all file names in the folder
file_names = os.listdir(mosaic_folder)

# Define the output file path
output_file = 'tile_names.txt'

# Write the file names to the output file
with open(output_file, 'w') as file:
    for file_name in file_names:
        file.write(file_name[:5] + '\n')