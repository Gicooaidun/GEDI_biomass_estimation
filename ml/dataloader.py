"""

This script defines the dataset class for the GEDI dataset.

"""

############################################################################################################################
# IMPORTS

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join
import pickle
from os.path import join, exists
import argparse
np.seterr(divide = 'ignore') 

# Define the nodata values for each data source
NODATAVALS = {'S2_bands' : 0, 'BM': -9999.0}

############################################################################################################################
# Helper functions

def initialize_index(fnames, mode, chunk_size, path_mapping, path_h5) :
    """
    This function creates the index for the dataset. The index is a dictionary which maps the file
    names (`fnames`) to the tiles that are in the `mode` (train, val, test); and the tiles to the
    number of chunks that make it up.

    Args:
    - fnames (list): list of file names
    - mode (str): the mode of the dataset (train, val, test)
    - chunk_size (int): the size of the chunks
    - path_mapping (str): the path to the file mapping each mode to its tiles

    Returns:
    - idx (dict): dictionary mapping the file names to the tiles and the tiles to the chunks
    - total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'mapping.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)

    # Iterate over all files
    idx = {}
    for fname in fnames :
        idx[fname] = {}
        
        with h5py.File(join(path_h5, fname), 'r') as f:
            
            # Get the tiles in this file which belong to the mode
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping[mode])
            
            # Iterate over the tiles
            for tile in tiles :

                # Get the number of patches in the tile
                n_patches = len(f[tile]['GEDI']['agbd'])
                idx[fname][tile] = n_patches // chunk_size
    
    total_length = sum(sum(v for v in d.values()) for d in idx.values())

    return idx, total_length
    return


def find_index_for_chunk(index, n, total_length):
    """
    For a given `index` and `n`-th chunk, find the file, tile, and row index corresponding
    to this chunk.
    
    Args:
    - index (dict): dictionary mapping the files to the tiles and the tiles to the chunks
    - n (int): the n-th chunk

    Returns:
    - file_name (str): the name of the file
    - tile_name (str): the name of the tile
    - chunk_within_tile (int): the chunk index within the tile
    """

    # Check that the chunk index is within bounds
    assert n < total_length, "The chunk index is out of bounds"

    # Iterate over the index to find the file, tile, and row index
    cumulative_sum = 0
    for file_name, file_data in index.items():
        for tile_name, num_rows in file_data.items():
            if cumulative_sum + num_rows > n:
                # Calculate the row index within the tile
                chunk_within_tile = n - cumulative_sum
                return file_name, tile_name, chunk_within_tile
            cumulative_sum += num_rows


def encode_lat_lon(lat, lon) :
    """
    Encode the latitude and longitude into sin/cosine values. We use a simple WRAP positional encoding, as 
    Mac Aodha et al. (2019).

    Args:
    - lat (float): the latitude
    - lon (float): the longitude

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # The latitude goes from -90 to 90
    lat_cos, lat_sin = np.cos(np.pi * lat / 90), np.sin(np.pi * lat / 90)
    # The longitude goes from -180 to 180
    lon_cos, lon_sin = np.cos(np.pi * lon / 180), np.sin(np.pi * lon / 180)

    # Now we put everything in the [0,1] range
    lat_cos, lat_sin = (lat_cos + 1) / 2, (lat_sin + 1) / 2
    lon_cos, lon_sin = (lon_cos + 1) / 2, (lon_sin + 1) / 2

    return lat_cos, lat_sin, lon_cos, lon_sin


def encode_coords(central_lat, central_lon, patch_size, resolution = 10) :
    """ 
    This function computes the latitude and longitude of a patch, from the latitude and longitude of its central pixel.
    It then encodes these values into sin/cosine values, and scales the results to [0,1].

    Args:
    - central_lat (float): the latitude of the central pixel
    - central_lon (float): the longitude of the central pixel
    - patch_size (tuple): the size of the patch
    - resolution (int): the resolution of the patch

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # Initialize arrays to store latitude and longitude coordinates

    i_indices, j_indices = np.indices(patch_size)

    # Calculate the distance offset in meters for each pixel
    offset_lat = (i_indices - patch_size[0] // 2) * resolution
    offset_lon = (j_indices - patch_size[1] // 2) * resolution

    # Calculate the latitude and longitude for each pixel
    latitudes = central_lat + (offset_lat / 6371000) * (180 / np.pi)
    longitudes = central_lon + (offset_lon / 6371000) * (180 / np.pi) / np.cos(central_lat * np.pi / 180)

    lat_cos, lat_sin, lon_cos, lon_sin = encode_lat_lon(latitudes, longitudes)

    return lat_cos, lat_sin, lon_cos, lon_sin


def normalize_data(data, norm_values, norm_strat, nodata_value = None) :
    """
    Normalize the data, according to various strategies:
    - mean_std: subtract the mean and divide by the standard deviation
    - pct: subtract the 1st percentile and divide by the 99th percentile
    - min_max: subtract the minimum and divide by the maximum

    Args:
    - data (np.array): the data to normalize
    - norm_values (dict): the normalization values
    - norm_strat (str): the normalization strategy

    Returns:
    - normalized_data (np.array): the normalized data
    """

    if norm_strat == 'mean_std' :
        mean, std = norm_values['mean'], norm_values['std']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - mean) / std)
        else : data = (data - mean) / std

    elif norm_strat == 'pct' :
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else :
            data = (data - p1) / (p99 - p1)
        data = np.clip(data, 0, 1)

    elif norm_strat == 'min_max' :
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)
    
    else: 
        raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')

    return data


def normalize_bands(bands_data, norm_values, order, norm_strat, nodata_value = None) :
    """
    This function normalizes the bands data using the normalization values and strategy.

    Args:
    - bands_data (np.array): the bands data to normalize
    - norm_values (dict): the normalization values
    - order (list): the order of the bands
    - norm_strat (str): the normalization strategy
    - nodata_value (int/float): the nodata value

    Returns:
    - bands_data (np.array): the normalized bands data
    """
    
    for i, band in enumerate(order) :
        band_norm = norm_values[band]
        bands_data[:, :, i] = normalize_data(bands_data[:, :, i], band_norm, norm_strat, nodata_value)
    
    return bands_data


class GEDIDataset(Dataset):

    def __init__(self, paths, fnames, chunk_size, mode, args):

        self.h5_path, self.norm_path, self.mapping = paths['h5'], paths['norm'], paths['map']
        self.mode = mode
        self.chunk_size = chunk_size
        self.index, self.length = initialize_index(fnames, self.mode, self.chunk_size, self.mapping, self.h5_path)

        # Define the data to use
        self.latlon = args.latlon
        self.bands = args.bands
        self.bm = args.bm # TODO changed
        self.patch_size = args.patch_size

        # Define the learning procedure
        self.norm_strat = args.norm_strat
        self.norm_target = args.norm

        # Check that the mode is valid
        assert self.mode in ['train', 'val', 'test'], "The mode must be one of 'train', 'val', 'test'"
        if not exists(join(self.norm_path, 'normalization_values.pkl')):
            raise FileNotFoundError('The file `normalization_values.pkl` does not exist.')

        # Load the normalization values
        with open(join(self.norm_path, 'normalization_values.pkl'), mode = 'rb') as f:
            self.norm_values = pickle.load(f)
        
        # Open the file handles
        self.handles = {fname: h5py.File(join(self.h5_path, fname), 'r') for fname in self.index.keys()}

        # Define the window size
        self.center = 7 # works if the patches are 15x15, otherwise, need to change
        self.window_size = self.patch_size[0] // 2
        

    def __len__(self):
        return self.length
    
    def __getitem__(self, n):
            
        # Find the file, tile, and row index corresponding to this chunk

        file_name, tile_name, idx = find_index_for_chunk(self.index, n, self.length)

        # Get the file handle
        f = self.handles[file_name]

        # Set the order and indices for the Sentinel-2 bands
        if not hasattr(self, 's2_indices') : self.s2_order = list(f[tile_name]['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.bands]
        if self.s2_order!= ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
            print('problem')

        data = []

        # Sentinel-2 bands
        if self.bands != [] :
            
            # Get the bands
            s2_bands = f[tile_name]['S2_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)

            # Get the BOA offset, if it exists
            if 'S2_boa_offset' in f[tile_name]['Sentinel_metadata'].keys() : 
                s2_boa_offset = f[tile_name]['Sentinel_metadata']['S2_boa_offset'][idx]
                print("s2_boa_offset")
            else: s2_boa_offset = 0

            # Get the surface reflectance values
            sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
            sr_bands[s2_bands == 0] = 0
            sr_bands[sr_bands < 0] = 0
            s2_bands = sr_bands

            s2_bands = normalize_bands(s2_bands, self.norm_values['S2_bands'], self.s2_order, self.norm_strat, NODATAVALS['S2_bands'])
            s2_bands = s2_bands[:, :, self.s2_indices]

            data.extend([s2_bands])
            

        # Latitude and longitude data
        lat_offset, lat_decimal = f[tile_name]['GEDI']['lat_offset'][idx], f[tile_name]['GEDI']['lat_decimal'][idx]
        lon_offset, lon_decimal = f[tile_name]['GEDI']['lon_offset'][idx], f[tile_name]['GEDI']['lon_decimal'][idx]
        lat, lon = lat_offset + lat_decimal, lon_offset + lon_decimal
        lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat, lon, self.patch_size)
        if self.latlon : data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
        else: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])
        
        # BM data
        if self.bm: # TODO done
            bm = f[tile_name]['BM']['bm'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            bm = normalize_data(bm, self.norm_values['BM']['bm'], self.norm_strat, NODATAVALS['BM'])
            
            bm_std = f[tile_name]['BM']['std'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            bm_std = normalize_data(bm_std, self.norm_values['BM']['std'], self.norm_strat, NODATAVALS['BM'])

            data.extend([bm[..., np.newaxis], bm_std[..., np.newaxis]])
        # Concatenate the data together
        data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 0)).to(torch.float)

        # Get the GEDI target data
        agbd = f[tile_name]['GEDI']['agbd'][idx]
        if self.norm_target :
            agbd = normalize_data(agbd, self.norm_values['GEDI']['agbd'], self.norm_strat)
        agbd = torch.from_numpy(np.array(agbd, dtype = np.float32)).to(torch.float)

        return data, agbd


############################################################################################################################
# Execute

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.latlon = True
    args.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    args.bm = True # TODO changed
    args.patch_size = [15,15]
    args.norm_strat = 'pct'
    args.norm = False

    fnames = ['data_nonan_0-5.h5', 'data_nonan_1-5.h5', 'data_nonan_2-5.h5', 'data_nonan_3-5.h5', 'data_nonan_4-5.h5'] # TODO changed
    
    for mode in ['train', 'val', 'test'] :
        print('Processing {} data...'.format(mode))
        
         # TODO CHANGE THESE
        ds = GEDIDataset({'h5':'/scratch2/biomass_estimation/code/ml/data', 'norm': '/scratch2/biomass_estimation/code/ml/data', 'map': '//scratch2/biomass_estimation/code/ml/data/'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)

        # Create a DataLoader instance
        data_loader = DataLoader(dataset = ds,
                                batch_size = 512,
                                shuffle = False,
                                num_workers = 8)

        # Iterate through the DataLoader

        print('starting to iterate...')
        i = 0
        for batch_samples in data_loader:
            images, targets = batch_samples
            # if i == 0 : 
                
            #     # print(images)
            #     # print(targets)
            # i += 1
            
            
            # Check for NaN values
            if torch.isnan(images).any() : 
                print('Data is NaN')
            
            # CHeck for inf values
            if torch.isinf(images).any() : 
                print('Data is inf')
            
            # Check that data is in [0,1] range
            if torch.min(images) < 0 or torch.max(images) > 1 : 
                print('Data is not in [0,1] range')
        
        print('done!')