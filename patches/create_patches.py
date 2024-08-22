"""

This script extracts patches of Sentinel-2 L2A data, BM data, and GEDI data, for the purpose of training a neural network to
predict biomass. 

Execution:
    python create_patches.py    --tilenames /path/to/tile_names.txt
                                --year 2019
                                --patch_size 15 15 
                                --chunk_size 1
                                --path_shp /path/to/sentinel_2_index_shapefile.shp 
                                --path_gedi /path/to/GEDI 
                                --path_s2 /path/to/S2_L2A 
                                --path_bm /path/to/BM
                                --output_path /path/to/patches
                                --output_fname (optional)
                                --BM
                                --i
                                --N


python3 create_patches.py --year 2019 --patch_size 15 15 --chunk_size 1 --output_fname "test" --BM --i 20 --N 50 --tilenames /scratch2/biomass_estimation/code/dataset_creation/tile_names.txt



"""

############################################################################################################################
# IMPORTS

import h5py
import glob
import argparse
import geopandas as gpd
from shutil import rmtree
from zipfile import ZipFile
from shapely.geometry import box
from os.path import join, basename, exists
import numpy as np

from helper_patches import *

# Absolute path to `S2_tiles_Siberia_all.geojson` file
local_path_shp = join('/scratch3', 'biomass_estimation_dominik_senti', 'github', 'data', 'S2_tiles_Siberia_polybox', 'S2_tiles_Siberia_all.geojson')

############################################################################################################################
# Helper functions

def setup_parser() :
    """ 
    Setup the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    # Arguments for the patches extraction procedure
    parser.add_argument('--year', help = 'Year of the data to process product.', type = str, required = True)
    parser.add_argument('--patch_size', help = 'Size of the patches to extract, in pixels.', nargs = 2, type = int, default = [15, 15])
    parser.add_argument('--chunk_size', help = 'Number of patches to write to file at once.', type = int, default = 1)
    parser.add_argument('--i', help = 'Process split i/N.', type = int, required = True)
    parser.add_argument('--N', help = 'Total number of splits.', type = int, required = True)

    # Paths arguments
    parser.add_argument('--tilenames', help = 'Path to a .txt file listing the S2 tiles to consider.')
    parser.add_argument('--path_shp', help = 'Path to the Sentinel-2 index shapefile.', default = local_path_shp)
    parser.add_argument('--path_gedi', help = 'Path to the GEDI data directory.', default = '/scratch3/biomass_estimation_dominik_senti/github/data/GEDI_data/L4A_Siberia.shp')
    parser.add_argument('--path_s2', help = 'Path to the Sentinel-2 data directory.', default = '/scratch3/Siberia')
    parser.add_argument('--path_bm', help = 'Path to the BM (ICESat) data.', default = '/scratch3/biomass_estimation_dominik_senti/github/data_preprocessing/cropped_mosaic')
    parser.add_argument('--output_path', help = 'Path to the output directory.', default = '/scratch2/biomass_estimation/temp')
    parser.add_argument('--output_fname', help = 'Name of the output file.', default = '')

    # Flags for the data to extract
    parser.add_argument('--BM', help = 'Whether to extract ICESat biomass patches.', action = 'store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Check that the tilenames argument is a .txt file
    if args.tilenames is not None: 
        if not args.tilenames.endswith('.txt'): 
            raise ValueError('Please provide a .txt file for the --tilenames argument.')
    
    # Check that the patch_size is odd
    if args.patch_size[0] % 2 == 0 or args.patch_size[1] % 2 == 0 :
        raise ValueError('patch_size must be odd.')

    return args.tilenames, args.year, args.patch_size, args.chunk_size, args.path_shp, args.path_gedi, args.path_s2, args.path_bm, args.output_path, args.output_fname, args.i, args.N, args.BM


def list_s2_tiles(tilenames, grid_df, path_s2) :
    # print(f'Listing S2 tiles from {path_s2}.')
    # print(f'Using shapefile {grid_df}.')
    # print(grid_df["geometry"].values[0])
    # print(f'Using tilenames file {tilenames}.')
    """
    This function performs two tasks: 1) return the list of Sentinel-2 tile names for which we want to extract patches (this
    is done either by listing the files in the Sentinel-2 data directory, or by reading a .txt file if one is provided); and
    2) return the geometries of those tiles, from the Sentinel-2 grid shapefile.

    Args:
    - tilenames: string, path to a .txt file listing the Sentinel-2 tiles to consider.
    - grid_df: geopandas dataframe, Sentinel-2 grid shapefile.

    Returns:
    - tile_names: list of strings, names of the Sentinel-2 tiles to consider.
    - tile_geoms: list of shapely geometries, geometries of the Sentinel-2 tiles to consider.
    """
    
    # Option 1 : list them from the folder of downloaded tiles
    if tilenames is None: 
        all_files = glob.glob(join(path_s2, f'*MSI*.zip'))
        # tile_names = [basename(f).strip('.zip') for f in all_files] 
        tile_names = [basename(f).strip('.zip')[39:44] for f in all_files]
    
    # Option 2 : list them from the .txt file
    else:
        with open(tilenames) as f: 
            tile_names = [tile_name.strip().strip('.zip') for tile_name in f.readlines()]
    
    # Get the geometries from the Sentinel-2 grid shapefile
    # tile_geoms = [grid_df[grid_df['Name'] == tile_name]["geometry"].values[0] for tile_name in tile_names] 
    tile_geoms = [grid_df[grid_df['Name'] == tile_name]["geometry"].values[0] for tile_name in tile_names]
    # print(f'Found {len(tile_geoms)} tiles to process.')
    # print(tile_geoms)
    return tile_names, tile_geoms



############################################################################################################################
# Main function

def extract_patches(tile_name, year, tile_geom, patch_size, chunk_size, path_gedi, path_s2, path_bm, output_path, output_fname, i, N, BM_flag) :
    """
    This function extracts the GEDI footprint-centered (patch_size[0] x patch_size[1]) patches from the Sentinel-2 L2A products with `tile_name`, as well
    as the corresponding BM data. The patches are iteratively saved to an hdf5 file.
    """

    # Load the GEDI data using the geometry of the tile, and reproject it to the tile's CRS
    print(f'Loading GEDI data for tile {tile_name}.')
    GEDI, crs_1 = load_GEDI_data(path_gedi, tile_geom, tile_name)
    if GEDI.empty :
        print(f'No GEDI data for tile {tile_name}.')
        print(f'Done for tile {tile_name}!')
        return

    # Group the footprints by their corresponding Sentinel-2 product
    print(f'Grouping GEDI data by Sentinel-2 product.')
    groups = group_GEDI_by_S2(GEDI, tile_name, path_s2)
    if list(groups) == [] :
        print(f'No S2 match for tile {tile_name}.')
        print(f'Done for tile {tile_name}!')
        return

    # Load the BM data for this year and tile
    if BM_flag: bm_raw = load_BM_data(path_bm, tile_name)

    # Name the output file
    if output_fname == '' : fname = f'data_{i}-{N}.h5'
    else: fname = f'data_{output_fname}_{i}-{N}.h5'
    
    # Open the output file
    with h5py.File(join(output_path, fname), 'a') as file :

        # Initialize the h5py group for the current tile
        print(f'Initializing output file for tile {tile_name}.')
        init_h5_group(file, tile_name, patch_size, chunk_size, BM_flag)

        # Iterate over the footprints with the same Sentinel-2 product
        print(f'Extracting patches for tile {tile_name}.')
        for s2_prod, footprints in groups :

            print(f'>> Extracting patches for product {s2_prod}.')
            # print(f'>> {len(footprints)} footprints to process.')
            # print(footprints.head())

            # Unzip the S2 L2A product if it hasn't been done 
            unzipped_path = join(path_s2,'scratch2','gsialelli','S2_L2A', 'Siberia')
            if not exists(join(unzipped_path, s2_prod + '.SAFE')) and not exists(join(path_s2, s2_prod + '.SAFE')):
                try:
                    with ZipFile(join(path_s2, s2_prod + '.zip'), 'r') as zip_ref:
                        zip_ref.extractall(path_s2)
                except Exception as e:
                    print(f'>> Could not unzip {s2_prod}.')
                    print(e)
                    continue
            if exists(join(path_s2, s2_prod + '.SAFE')) :
                unzipped_path = path_s2

            # Reproject and upsample the S2 bands            
            try: 
                transform, upsampling_shape, processed_bands, crs_2, bounds = process_S2_tile(s2_prod, path_s2 = unzipped_path)
            except IndexError:
                unzipped_path = join(path_s2, 'scratch2', 'gsialelli', 'S2_L2A', 'Siberia')
                try:
                    transform, upsampling_shape, processed_bands, crs_2, bounds = process_S2_tile(s2_prod, path_s2 = unzipped_path)
                except Exception as e:
                    print(f'>> Could not process product {s2_prod}.')
                    print(e)
                    continue
            except Exception as e:
                print(f'>> Could not process product {s2_prod}.')
                print(e)
                continue
            
            assert crs_1 == crs_2 == footprints.crs, "CRS mismatch."


            # Process the BM tile corresponding to the S2 product # TODO modify
            if BM_flag: bm_tile = get_tile(bm_raw, transform, upsampling_shape, 'BM', BM_attrs)

            # Initialize results placeholder
            s2_data, gedi_data, bm_data = initialize_results(BM_flag)

            # Further crop the data to the product's bounds
            footprints = footprints[footprints.intersects(box(*bounds))]

            # Iterate over the footprints
            for footprint in footprints.itertuples() :

                # Extract the Sentinel-2 data
                # print('transform in create_patches')
                # print(transform)
                s2_footprint_data = get_sentinel2_patch(transform, processed_bands, footprint, patch_size, s2_prod)
                if s2_footprint_data is None: continue

                # Extract the BM data # TODO modify
                if BM_flag: bm_footprint_data = get_patch(bm_tile, footprint, transform, patch_size, 'BM', BM_attrs)
                else: bm_footprint_data = None

                # Extract the GEDI data
                gedi_footprint_data = get_gedi_data(footprint)

                # Aggregate the results
                s2_data, gedi_data, bm_data = update_results(s2_data, gedi_data, bm_data, s2_footprint_data, gedi_footprint_data, bm_footprint_data)

                # Write the results to file and reset the placeholders
                num_patches = len(gedi_data['agbd'])

                #outlier exclusion
                # print("legth of gedi_data[agbd]", len(gedi_data['agbd']))
                # print(gedi_data['agbd'])
                if gedi_data['agbd'][0] < 500:
                    if (num_patches % chunk_size) == 0 :
                        save_results(s2_data, gedi_data, bm_data, tile_name, chunk_size, file)
                        s2_data, gedi_data, bm_data = initialize_results(BM_flag)

            # Remove the unzipped S2 product
            rmtree(join(unzipped_path, s2_prod + '.SAFE'))

    print(f'Done for tile {tile_name}!')

############################################################################################################################
# Execute

import time

if __name__ == "__main__":

    # Parse the command line arguments
    tilenames, year, patch_size, chunk_size, path_shp, path_gedi, path_s2, path_bm, output_path, output_fname, i, N, BM_flag = setup_parser()

    # Read the Sentinel-2 grid shapefile
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio')

    # List all S2 tiles and their geometries
    tile_names, tile_geoms = list_s2_tiles(tilenames, grid_df, path_s2)

    # Split into N, and process the i-th split
    tile_names = np.array_split(tile_names, N)[i]
    tile_geoms = np.array_split(tile_geoms, N)[i]
    assert len(tile_names) == len(tile_geoms)

    setup_output_files(output_path, output_fname, i, N)

    start_time = time.time()

    for tile_name, tile_geom in zip(tile_names, tile_geoms) :
        try: 
            extract_patches(tile_name, year, tile_geom, patch_size, chunk_size, path_gedi, path_s2, path_bm, output_path, output_fname, i, N, BM_flag)
        except Exception as e: 
            print(f"Couldn't extract patches for tile {tile_name}.", e)
            continue

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds.')
