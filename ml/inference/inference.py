import argparse
import os
import glob
import sys
import traceback
import geopandas as gpd
import numpy as np
from zipfile import ZipFile
import pickle
from os.path import join
from shutil import rmtree
import torch.nn as nn
from affine import Affine
from pyproj import Transformer
import torch
import cv2
sys.path.insert(1, '../../patches')
from helper_patches import *
from create_patches import *
import h5py
import matplotlib.pyplot as plt

def setup_parser():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--path_icesat", type=str, default="data_preprocessing/cropped_mosaic")
    parser.add_argument("--tilenames", type=str, default="tile_names_inference_all.txt")
    parser.add_argument("--path_shp", type=str, default=os.path.join('..', '..', 'data', 'S2_tiles_Siberia_polybox', 'S2_tiles_Siberia_all.geojson'))
    parser.add_argument("--path_s2", type=str, default="/scratch3/Siberia")
    parser.add_argument("--norm_path", type=str, default="../dataset/normalization_values.pkl")
    parser.add_argument("--model_path", type=str, default="../models")
    parser.add_argument("--patch_size", type=int, default=277)

    args = parser.parse_args()
    return args.path_shp, args.tilenames, args.path_s2, args.path_icesat, args.norm_path, args.model_path, args.patch_size

class SimpleFCN(nn.Module):
    def __init__(self,
                 in_features=18,
                 channel_dims = (16, 32, 64, 128, 64, 32, 16),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1):
        """
        A simple fully convolutional neural network.
        """
        super(SimpleFCN, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        layers = list()
        for i in range(len(channel_dims)):
            in_channels = in_features if i == 0 else channel_dims[i-1]
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=channel_dims[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1, padding_mode='reflect'))
            layers.append(nn.BatchNorm2d(num_features=channel_dims[i]))
            layers.append(self.relu)
        # print(layers)
        self.conv_layers = nn.Sequential(*layers)
        
        self.conv_output = nn.Conv2d(in_channels=channel_dims[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_output(x)

        return x
    
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        if torch.cuda.is_available():
            self.models = [model.cuda() for model in self.models]
        outputs = [model(x) for model in self.models]
        mean = torch.mean(torch.stack(outputs), dim=0)
        std = torch.std(torch.stack(outputs), dim=0)
        return mean, std
    
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
    normalized = {}
    for i, band in enumerate(order) :
        if band != 'SCL' and band != 'transform':
            band_norm = norm_values[band]
            # print(band_norm)
            # print(bands_data[band].shape)
            normalized[band] = normalize_data(bands_data[band], band_norm, norm_strat, nodata_value)
    
    return normalized

def load_files(s2_prod, path_s2, path_icesat):
    """
    This function processes Sentinel-2 (S2) products by extracting, unzipping, reprojecting, and upsampling the data. 
    It also loads corresponding ICESat-2 biomass data. 

    Args:
    - s2_prod (str): the Sentinel-2 product identifier to process.
    - path_s2 (str): the path to the directory containing the S2 data.
    - path_icesat (str): the path to the directory containing the ICESat-2 data.

    Returns:
    - s2_processed_bands (list): a list containing the processed S2 bands data.
    - s2_transforms (list): a list of transformation matrices for the S2 data.
    - s2_crs (list): a list of coordinate reference systems (CRS) for the S2 data.
    - icesat_raws (list): a list containing the raw ICESat-2 biomass data corresponding to the S2 product.
    """
    s2_processed_bands = []
    s2_transforms = []
    s2_crs = []
    icesat_raws = []

    print(f'>> Extracting patches for product {s2_prod}.')

    # unzip all zip files which contain the S2 tile name
    all_corresponding_s2_paths = glob.glob(f"{path_s2}/*{s2_prod}*")
    print(f'>> Found {all_corresponding_s2_paths}.')
    for total_s2_path in all_corresponding_s2_paths:
        if total_s2_path.endswith('.zip'):
                try:
                    with ZipFile(total_s2_path, 'r') as zip_ref:
                        zip_ref.extractall(path=total_s2_path[:-4])
                except Exception as e:
                    print(f'>> Could not unzip {total_s2_path}.')
                    print(e)
                    continue

    # for the unzipped files, find the .SAFE folder and process the tiles
    all_corresponding_s2_paths = glob.glob(f"{path_s2}/*{s2_prod}*")

    for total_s2_path in all_corresponding_s2_paths:
        if total_s2_path.endswith('.zip'):
            continue
        print(f'>> Processing {total_s2_path}.')

        if os.path.isdir(total_s2_path):
            subfolders = [f.path for f in os.scandir(total_s2_path) if f.is_dir()]
            while subfolders:
                subfolder = subfolders.pop(0)
                if subfolder.endswith('.SAFE'):
                    total_s2_path = subfolder
                    total_unzipped_path = subfolder
                    break
                else:
                    subfolders.extend([f.path for f in os.scandir(subfolder) if f.is_dir()])

        # extract path and file name
        s2_folder_path, s2_file_name = os.path.split(total_s2_path)
        if s2_file_name.endswith('.SAFE'):
            s2_file_name = s2_file_name[:-5]
        else:
            print(f'>> {s2_file_name} does not end with .SAFE.')
            continue

        # Reproject and upsample the S2 bands            
        try: 
            transform, upsampling_shape, processed_bands, crs_2, bounds = process_S2_tile(product=s2_file_name, path_s2 = s2_folder_path)
        except Exception as e:
            print(f'>> Could not process product {s2_prod}.')
            print(traceback.format_exc())
            continue
        
        s2_processed_bands.append(processed_bands)
        s2_transforms.append(transform)
        s2_crs.append(crs_2)
    
    for total_s2_path in all_corresponding_s2_paths:
        if total_s2_path.endswith('.zip'):
            continue
        else:
            try:
                # Remove the unzipped S2 product
                rmtree(total_s2_path)
            except Exception as e:
                print(f'>> Could not remove {total_s2_path}.')
                print(e)
    
    icesat_raw = load_BM_data(path_bm=path_icesat, tile_name=s2_prod)
    icesat_raws.append(icesat_raw)

    return s2_processed_bands, s2_transforms, s2_crs, icesat_raws

def normalize(processed_bands, icesat_raw):
    """
    This function normalizes both the Sentinel-2 (S2) bands data and the ICESat-2 biomass data using pre-calculated
        normalization values. 

    Args:
    - processed_bands (list): a list of processed Sentinel-2 bands data to be normalized.
    - icesat_raw (list): a list of raw ICESat-2 biomass data to be normalized.

    Returns:
    - icesat_norm (list): a list containing the normalized ICESat-2 biomass data.
    - s2_bands_dict (list): a list of dictionaries containing the normalized S2 bands data.
    - s2_indices (list): a list of lists containing the indices of the bands in the order they were processed.
    """
    icesat_norm = []
    s2_bands_dict = []
    s2_indices = []
    s2_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    with open(norm_path, mode = 'rb') as f:
                norm_values = pickle.load(f)

    norm_strat = "pct"
    for i in range(len(processed_bands)):
        icesat_temp = {}
        icesat_temp['bm'] = normalize_data(icesat_raw[0]['bm'], norm_values['BM']['bm'], norm_strat, nodata_value = -9999.0)
        icesat_temp['std'] = normalize_data(icesat_raw[0]['std'], norm_values['BM']['std'], norm_strat, nodata_value = -9999.0)
        icesat_norm.append(icesat_temp)

        s2_bands_dict.append(normalize_bands(processed_bands[i], norm_values['S2_bands'], s2_order, norm_strat, nodata_value = 0))
        s2_indices.append([s2_order.index(band) for band in s2_bands_dict[i]])
    return icesat_norm, s2_bands_dict, s2_indices

#######################################################
# weight matrix functions
def calculate_central_weight_matrix(patch):
    """
    This function calculates a central weight matrix for a given patch, 
    where the weights are determined by the distance of each point in the patch to the central four quadrants.
    The closer a point is to the center, the higher its weight, with weights normalized across each quadrant.

    Args:
    - patch (np.array): a 2D numpy array representing the patch for which the weight matrix is to be calculated. 
        The patch is assumed to be square-shaped.

    Returns:
    - weight_matrix (np.array): a 2D numpy array of the same shape as the input patch, containing the calculated 
        weights for each point in the patch.
    """
    sizeOfPatch = patch.shape[0]
    half_patch_size = sizeOfPatch // 2
    distances = np.zeros((half_patch_size, half_patch_size, 4))
    
    for i in range(half_patch_size):
        for j in range(half_patch_size):
            distances[i, j, 0] = np.sqrt(i**2 + j**2)  # Top-left corner
            distances[i, j, 1] = np.sqrt(i**2 + (half_patch_size - j - 1)**2)  # Top-right corner
            distances[i, j, 2] = np.sqrt((half_patch_size - i - 1)**2 + j**2)  # Bottom-left corner
            distances[i, j, 3] = np.sqrt((half_patch_size - i - 1)**2 + (half_patch_size - j - 1)**2)  # Bottom-right corner
    
    normalized_distances = np.max(distances) - distances
    normalized_distances = normalized_distances / np.sum(normalized_distances, axis=2, keepdims=True)
    
    weight_matrix = np.zeros((sizeOfPatch, sizeOfPatch))
    weight_matrix[:half_patch_size, :half_patch_size] = normalized_distances[:,:,3]
    weight_matrix[:half_patch_size, half_patch_size:sizeOfPatch] = normalized_distances[:,:,2]
    weight_matrix[half_patch_size:sizeOfPatch, :half_patch_size] = normalized_distances[:,:,1]
    weight_matrix[half_patch_size:sizeOfPatch, half_patch_size:sizeOfPatch] = normalized_distances[:,:,0]
    
    return weight_matrix

def calculate_edge_distances(patch):
    """
    This function calculates a matrix of distances from each point in a patch to the four edges (left, right, top, and bottom). 
    The distances are normalized and scaled, with closer points to the edges having higher weights.

    Args:
    - patch (np.array): a 2D numpy array representing the patch for which the edge distances are to be calculated. 
        The patch is assumed to be square-shaped.

    Returns:
    - normalized_distances (np.array): a 3D numpy array where the first two dimensions match the input patch, 
        and the third dimension contains the normalized and scaled distances to each of the four edges (left, right, top, bottom).
    """
    patch_size = patch.shape[0]
    distances = np.zeros((patch_size, patch_size, 4))
    
    for i in range(patch_size):
        for j in range(patch_size):
            distances[i, j, 0] = np.sqrt(j**2)  #horizontal edge left side
            distances[i, j, 1] = np.sqrt((patch_size - j - 1)**2)  #horizontal edge right side
            distances[i, j, 2] = np.sqrt(i**2)  #vertical edge top side
            distances[i, j, 3] = np.sqrt((patch_size - i - 1)**2)  #vertical edge bottom side

    normalized_distances = np.max(distances) - distances
    normalized_distances = normalized_distances / np.sum(normalized_distances, axis=2, keepdims=True)
    return normalized_distances*2

def calculate_weight_matrix(patch, corner = 0, edge = 0):
    """
    This function calculates a weight matrix for a given patch, combining central weights and edge 
    weights based on specified parameters. 
    The weights are adjusted depending on whether the focus is on a corner or an edge of the patch.

    Args:
    - patch (np.array): a 2D numpy array representing the patch for which the weight matrix is to be calculated.
    - corner (int): an integer specifying which corner of the patch should have full weight (1 by default). 
                If set to 0, edge weights are used instead. Possible values:
                0 - No corner emphasis, use edge weights
                1 - Top-left corner
                2 - Top-right corner
                3 - Bottom-right corner
                4 - Bottom-left corner
    - edge (int): an integer specifying which edge of the patch should have increased weight when no corner is emphasized. 
                This is only used if `corner` is set to 0. Possible values:
                0 - No edge emphasis
                1 - Left edge
                2 - Top edge
                3 - Right edge
                4 - Bottom edge

    Returns:
    - weight_matrix (np.array): a 2D numpy array containing the calculated weights for each point in the patch.
    """

    sizeOfPatch = patch.shape[0]
    half_patch_size = sizeOfPatch // 2
    
    # Calculate distances for the smaller half patch
    half_patch = np.zeros((half_patch_size, half_patch_size))
    edge_distances = calculate_edge_distances(half_patch)
    
    weight_matrix = calculate_central_weight_matrix(np.ones((sizeOfPatch, sizeOfPatch)))
    if corner == 0:
        if edge == 1:
            weight_matrix[:half_patch_size, half_patch_size:sizeOfPatch] = edge_distances[:, :, 0]
            weight_matrix[:half_patch_size, :half_patch_size] = edge_distances[:, :, 1]
        if edge == 2:
            weight_matrix[half_patch_size:sizeOfPatch, half_patch_size:sizeOfPatch] = edge_distances[:, :, 2]
            weight_matrix[:half_patch_size, half_patch_size:sizeOfPatch] = edge_distances[:, :, 3]
        if edge == 3:
            weight_matrix[half_patch_size:sizeOfPatch, half_patch_size:sizeOfPatch] = edge_distances[:, :, 0]
            weight_matrix[half_patch_size:sizeOfPatch, :half_patch_size] = edge_distances[:, :, 1]
        if edge == 4:
            weight_matrix[half_patch_size:sizeOfPatch, :half_patch_size] = edge_distances[:, :, 2]
            weight_matrix[:half_patch_size, :half_patch_size] = edge_distances[:, :, 3]
    elif corner == 1:
        weight_matrix[:half_patch_size, :half_patch_size] = 1
        weight_matrix[:half_patch_size, half_patch_size:sizeOfPatch] = edge_distances[:, :, 0]
        weight_matrix[half_patch_size:sizeOfPatch, :half_patch_size] = edge_distances[:, :, 2]
    elif corner == 2:
        weight_matrix[:half_patch_size, half_patch_size:sizeOfPatch] = 1
        weight_matrix[half_patch_size:sizeOfPatch, half_patch_size:sizeOfPatch] = edge_distances[:, :, 2]
        weight_matrix[:half_patch_size, :half_patch_size] = edge_distances[:, :, 1]
    elif corner == 3:
        weight_matrix[half_patch_size:sizeOfPatch, half_patch_size:sizeOfPatch] = 1
        weight_matrix[half_patch_size:sizeOfPatch, :half_patch_size] = edge_distances[:, :, 1]
        weight_matrix[:half_patch_size, half_patch_size:sizeOfPatch] = edge_distances[:, :, 3]
    elif corner == 4:
        weight_matrix[half_patch_size:sizeOfPatch, :half_patch_size] = 1
        weight_matrix[half_patch_size:sizeOfPatch, half_patch_size:sizeOfPatch] = edge_distances[:, :, 0]
        weight_matrix[:half_patch_size, :half_patch_size] = edge_distances[:, :, 3]

    return weight_matrix
# weight matrix functions
#######################################################


if __name__ == "__main__":

    #get arguments
    path_shp, tilenames, path_s2, path_icesat, norm_path, model_path, patch_size = setup_parser()
    center = patch_size // 2

    # Create a weight matrix
    print(">>> Creating weight matrix")
    sample_patch = np.random.rand(1000, 1000)
    central_weight_matrix = calculate_weight_matrix(sample_patch)
    ul_corner_weight_matrix = calculate_weight_matrix(sample_patch, corner = 1)
    ll_corner_weight_matrix = calculate_weight_matrix(sample_patch, corner = 2)
    lr_corner_weight_matrix = calculate_weight_matrix(sample_patch, corner = 3)
    ur_corner_weight_matrix = calculate_weight_matrix(sample_patch, corner = 4)
    left_edge_weight_matrix = calculate_weight_matrix(sample_patch, edge = 1)
    bottom_edge_weight_matrix = calculate_weight_matrix(sample_patch, edge = 2)
    right_edge_weight_matrix = calculate_weight_matrix(sample_patch, edge = 3)
    top_edge_weight_matrix = calculate_weight_matrix(sample_patch, edge = 4)

    central_weight_matrix = cv2.resize(central_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    central_weight_matrix[center, :] += central_weight_matrix[0,:]
    central_weight_matrix[:, center] += central_weight_matrix[:, 0]

    ul_corner_weight_matrix = cv2.resize(ul_corner_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    ul_corner_weight_matrix[center:, center] += 0.5*central_weight_matrix[center:, 0]
    ul_corner_weight_matrix[center, center:] += 0.5*central_weight_matrix[0, center:]

    ll_corner_weight_matrix = cv2.resize(ll_corner_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    ll_corner_weight_matrix[center, :center] += 0.5*central_weight_matrix[0, :center]
    ll_corner_weight_matrix[center:, center] += 0.5*central_weight_matrix[center:, 0]

    lr_corner_weight_matrix = cv2.resize(lr_corner_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    lr_corner_weight_matrix[:center, center] += 0.5*central_weight_matrix[:center, 0]
    lr_corner_weight_matrix[center, :center] += 0.5*central_weight_matrix[0, :center]

    ur_corner_weight_matrix = cv2.resize(ur_corner_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    ur_corner_weight_matrix[center, center:] += 0.5*central_weight_matrix[0, center:]
    ur_corner_weight_matrix[:center, center] += 0.5*central_weight_matrix[:center, 0]

    left_edge_weight_matrix = cv2.resize(left_edge_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    left_edge_weight_matrix[center, :] += 0.5*central_weight_matrix[0,:]
    left_edge_weight_matrix[center:, center] += central_weight_matrix[center:, 0]

    bottom_edge_weight_matrix = cv2.resize(bottom_edge_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    bottom_edge_weight_matrix[:, center] += 0.5*central_weight_matrix[:, 0]
    bottom_edge_weight_matrix[center, :center] += central_weight_matrix[0, :center]

    right_edge_weight_matrix = cv2.resize(right_edge_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    right_edge_weight_matrix[center, :] += 0.5*central_weight_matrix[0,:]
    right_edge_weight_matrix[:center, center] += central_weight_matrix[0,:center]

    top_edge_weight_matrix = cv2.resize(top_edge_weight_matrix, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    top_edge_weight_matrix[:, center] += 0.5*central_weight_matrix[:, 0]
    top_edge_weight_matrix[center, center:] += central_weight_matrix[0, center:]
    
    print(">>> reading tile names")
    # Read the Sentinel-2 grid shapefile
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio')
    # List all S2 tiles and their geometries
    tile_names, tile_geoms = list_s2_tiles(tilenames, grid_df, path_s2)

    print(">>> loading models")
    sub_models = []
    with torch.inference_mode():
        for i in range(1):
            temp_model = SimpleFCN()
            temp_model.load_state_dict(torch.load(os.path.join(model_path, f'balanced_icesat_100Epochs_LR0.01_Adam_seed24.pth')))
            temp_model.eval()
            sub_models.append(temp_model)
        
        model = EnsembleModel(sub_models)

    for s2_prod in tile_names:
        print(">>> starting tile ", s2_prod)
        #load s2 and icesat data
        s2_processed_bands, s2_transforms, s2_crs, icesat_raws = load_files(s2_prod, path_s2, path_icesat)
        print(">>> loaded files, starting normalization")
        #normalize the data with the normalization values
        icesat_norm, s2_bands_dict, s2_indices = normalize(s2_processed_bands, icesat_raws)

        #make np arrays from dicts
        s2_bands = []
        for i in range(len(s2_bands_dict)):
            temp = np.stack([s2_bands_dict[i][key] for key in s2_bands_dict[i].keys()], axis=-1)
            temp = temp[:, :, s2_indices[i]]
            s2_bands.append(temp)

        print(">>> normalized data, starting upsamling icesat data and loading models")

        upsampled_icesat = []
        for i in range(len(icesat_norm)):
            icesat = {}
            icesat['bm'] = upsampling_with_nans(icesat_norm[i]['bm'], s2_bands_dict[i]['B01'].shape, -9999, 3)
            icesat['std'] = upsampling_with_nans(icesat_norm[i]['std'], s2_bands_dict[i]['B01'].shape, -9999, 3)
            upsampled_icesat.append(icesat) 


        outputs = []
        
        print(">>> starting inference")
        for t in range(len(s2_bands)):
            fwd = Affine.from_gdal(s2_transforms[t][2], s2_transforms[t][0], s2_transforms[t][1], s2_transforms[t][5], s2_transforms[t][3], s2_transforms[t][4])
            coordinate_transformer = Transformer.from_crs(s2_crs[t], 'epsg:4326')
            outputs.append(np.zeros((icesat['bm'].shape[0], icesat['bm'].shape[1], 2)))

            for i in range(center, icesat['bm'].shape[0], center+1):
                for j in range(center, icesat['bm'].shape[1], center+1):
                    data = []
                    s2_temp = s2_bands[t][i-center:i+center+1, j-center:j+center+1,:]
                    data.extend([s2_temp])

                    lat1, lon1 = fwd * (i, j)
                    lat2, lon2 = coordinate_transformer.transform(lat1, lon1)
                    lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat2, lon2, (patch_size, patch_size))
                    data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])

                    icesat_temp_bm = upsampled_icesat[t]['bm'][i-center:i+center+1, j-center:j+center+1, np.newaxis]
                    icesat_temp_std = upsampled_icesat[t]['std'][i-center:i+center+1, j-center:j+center+1, np.newaxis]
                    data.extend([icesat_temp_bm, icesat_temp_std])

                    # Concatenate the data together
                    data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 0)).to(torch.float)
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            data = data.cuda()
                        result = model(data.unsqueeze(0))
                        result0 = result[0].detach().cpu().numpy().squeeze()
                        result1 = result[1].detach().cpu().numpy().squeeze()
                        
                        if j == center and i == center:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, ul_corner_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, ul_corner_weight_matrix)
                        elif i == center and j == icesat['bm'].shape[0] - center - 1:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, ur_corner_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, ur_corner_weight_matrix)
                        elif i == icesat['bm'].shape[1] - center - 1 and j == center:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, ll_corner_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, ll_corner_weight_matrix)
                        elif i == icesat['bm'].shape[1] - center - 1 and j == icesat['bm'].shape[0] - center - 1:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, lr_corner_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, lr_corner_weight_matrix)
                        elif i == center:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, top_edge_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, top_edge_weight_matrix)
                        elif i == icesat['bm'].shape[1] - center - 1:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, bottom_edge_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, bottom_edge_weight_matrix)
                        elif j == center:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, left_edge_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, left_edge_weight_matrix)
                        elif j == icesat['bm'].shape[0] - center - 1:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, right_edge_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, right_edge_weight_matrix)
                        else:
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 0] += np.multiply(result0, central_weight_matrix)
                            outputs[t][j-center:j+center+1, i-center:i+center+1, 1] += np.multiply(result1, central_weight_matrix)

            print('>>> Inference done for tile ', t+1)

        print("saving outputs for tile ", s2_prod)
        # Save the outputs as a .npy file
        outputs = np.moveaxis(outputs, 1, 2)
        np.save(f'inference_output/inference_{s2_prod}.npy', outputs)
        del outputs
        del s2_bands
        del icesat_norm
        del upsampled_icesat
        del s2_processed_bands
        del s2_transforms
        del s2_crs
        del icesat_raws
        del s2_bands_dict
        del icesat
        print('==============================')