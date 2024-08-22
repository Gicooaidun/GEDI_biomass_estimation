import geopandas as gpd
import pandas as pd
import numpy as np
import csv
import os
import rasterio as rs
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask


def custom_reproject(src_path, output_path, dst_crs):
    # Open the original raster dataset
    with rs.open(src_path) as src:
        
        # Calculate the transformation parameters and the width and height of the output
        # in the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Copy the metadata from the source dataset
        kwargs = src.meta.copy()
        # Update the metadata with the new CRS, transformation, and dimensions
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Specify the path to the merged_mosaic folder
        output_folder = '/scratch3/biomass_estimation_dominik_senti/github/statistics/reprojected_icesat_to_gedi_crs'

        # Check if the merged_mosaic folder exists
        if not os.path.exists(output_folder):
            # Create the merged_mosaic folder
            print(f'Creating the {output_folder} folder')
            os.makedirs(output_folder)

        # Open a new raster file for the reprojected data
        with rs.open(output_path, 'w', **kwargs) as dst:
            
            # Reproject each band in the raster dataset
            for i in range(1, src.count + 1):
                reproject(
                    # Source raster band
                    source=rs.band(src, i),
                    # Destination raster band
                    destination=rs.band(dst, i),
                    # Source transformation and CRS
                    src_transform=src.transform,
                    src_crs=src.crs,
                    # Destination transformation and CRS
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    # Resampling method
                    resampling=Resampling.nearest)


# Specify the path to the folder containing the tif files
folder_path = '/scratch3/biomass_estimation_dominik_senti/github/data/icesat_data'

# Iterate over the tif files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.tif'):
        print("processing: " + file_name)
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        dest_crs = 'EPSG:4326'

        
        # Perform the reproject on the tif file
        custom_reproject(file_path, f'/scratch3/biomass_estimation_dominik_senti/github/statistics/reprojected_icesat_to_gedi_crs/{file_name[:-4]}_reprojected.tif', dest_crs)