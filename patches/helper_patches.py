"""

Helper functions for `create_patches`

"""

import h5py
import glob
import geopandas as gpd
from shapely.geometry import box
from os.path import join, basename
import numpy as np
import geopandas as gpd
import numpy as np
import sys
import pandas as pd
import rasterio as rs
import datetime as dt
from rasterio.crs import CRS
import xml.etree.ElementTree as ET
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
from rasterio.transform import rowcol, AffineTransformer
sys.path.append('../Sentinel')
# from Sentinel_settings import GEDI_START_MISSION
GEDI_START_MISSION = '2019-01-01'

# Sentinel-2 L2A bands that we want to use
S2_L2A_BANDS = {'10m' : ['B02', 'B03', 'B04', 'B08'],
                '20m' : ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],
                '60m' : ['B01', 'B09']}

# GEDI attributes and their corresponding data types
GEDI_attrs = {
    # GEDI inherent attributes
    'pft_class': np.uint8, 'region_cla': np.uint8, 'leaf_off_f': np.uint8, 'urban_prop': np.uint8, 'agbd': np.float32,
       'agbd_se': np.float32, 'elev_lowes': np.float32, 'selected_a': np.uint8, 'shot_numbe': np.uint64, 
       'sensitivit': np.float32, 'solar_elev': np.float32, 'rh98': np.float32, 'date': np.uint16, 
    # GEDI derived attributes   
    'granule_num': np.uint8, 'track_num': np.uint32, 'ppds': np.uint8, 'version_num': np.uint16, 'lat_offset': np.uint8, 
        'lat_decimal': np.float32, 'lon_offset': np.uint8, 'lon_decimal': np.float32
    }

# Sentinel-2 attributes and their corresponding data types (we differentiate between 2d and 1d attributes)
S2_attrs = {'bands' : {'B01': np.uint16, 'B02': np.uint16, 'B03': np.uint16, 'B04': np.uint16, 'B05': np.uint16, 'B06': np.uint16, 
                        'B07': np.uint16, 'B08': np.uint16, 'B8A': np.uint16, 'B09': np.uint16, 'B11': np.uint16, 'B12': np.uint16, 
                        'SCL': np.uint8},
            'metadata' : {'vegetation_score': np.uint8, 'date' : np.int16, 'pbn' : np.uint16, 'ron' : np.uint8}
            }

# Biomass (BM) attributes and their corresponding data types
BM_attrs = {'bm': np.float32, 'std': np.float32}

NODATAVALS = {'S2' : 0, 'BM': -9999.0}


def explode_pattern(pattern) :
    """
    `pattern` (string) : the `date`, `orbit_number`, `granule_number`, `track_number`, `ppds_type` and `version_number`
            of the input file; parsed as explained in daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html
    cf. process_h5_filename() in GEDI/h5_to_csv_to_shp.py

    The date (YYYYDDDHHMMSS) don't keep because we have date, don't need H/M/S
    The orbit number don't keep because is in shot_number

    """
    _, _, granule_number, track_number, ppds_type, version_number = pattern.split('_')
    track_number, version_number = track_number.lstrip('T'), version_number.lstrip('V')
    return int(granule_number), int(track_number), int(ppds_type), int(version_number)


def explode_fp(x) :
    """
    For a floating point (double precision) number `x`, we first split it into its fractional and decimal parts. Then,
    we convert the fractional part to a single precision floating point number; and the decimal part to an unsigned 8-bit
    integer.

    Args:
    - x: float64, floating point number.

    Returns:
    - fractional: float32, fractional part of `x`.
    - decimal: uint8, decimal part of `x`.
    """
    fractional, decimal = np.modf(x)
    return fractional.astype(np.float32), np.abs(decimal).astype(np.uint8)


def modify_GEDI_data(GEDI) :
    """
    This function implements the following changes to the GEDI data:
        1) drop the following columns: 'doy_cos', 'doy_sin', 'lat_cos', 'lat_sin', 'lon_cos', 'lon_sin', 'beam';
        2) explode the `pattern` column into the following columns: 'granule_number', 'track_number', 'ppds_type', 
            'version_number'; and drop the `pattern` column;
        3) for the `lat_lowest` and `lon_lowest` columns, split them into `lat_offset`, `lat_decimal`, `lon_offset`,
            and `lon_decimal`, and drop the original columns.
    
    Args:
    - GEDI: geopandas dataframe, GEDI data.

    Returns:
    - GEDI: geopandas dataframe, modified GEDI data.
    """
    
    # 1) Remove columns
    columns_to_remove = ['beam'] #add other unnecessary columns
    """" 
    ['pft_class', 'region_cla', 'leaf_off_f', 'urban_prop', 'agbd',
       'agbd_se', 'beam', 'elev_lowes', 'lat_lowest', 'lon_lowest',
       'selected_a', 'shot_numbe', 'sensitivit', 'solar_elev', 'rh98',
       'pattern', 'date', 'geometry']
    """

    # 2) Explode the `pattern` column
    GEDI[['granule_num', 'track_num', 'ppds','version_num']] = GEDI['pattern'].apply(lambda x: pd.Series(explode_pattern(x)))
    columns_to_remove.append('pattern')

    # 3) Split the `lat_lowest` and `lon_lowest` columns
    GEDI[['lat_decimal', 'lat_offset']] = GEDI['lat_lowest'].apply(lambda x: pd.Series(explode_fp(x)))
    GEDI[['lon_decimal', 'lon_offset']] = GEDI['lon_lowest'].apply(lambda x: pd.Series(explode_fp(x)))

    # Specify the data types of the new columns
    GEDI = GEDI.astype(GEDI_attrs)
    
    return GEDI.drop(columns = columns_to_remove)


def get_CRS_from_S2_tilename(tname) :
    """
    Get the CRS of the Sentinel-2 tile from its name. The tiles are named as DDCCC (where D is a digit and C a character).
    MGRS tiles are in UTM projection, which means the CRS will be EPSG=326xx in the Northern Hemisphere, and 327xx in the
    Southern. The first character of the tile name gives you the hemisphere (C to M is South, N to X is North); and the
    two digits give you the UTM zone number.

    Args:
    - tname: str, name of the Sentinel-2 tile

    Returns:
    - rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    """

    tile_code, hemisphere = tname[:2], tname[2]

    if 'C' <= hemisphere <= 'M':
        crs = f'EPSG:327{tile_code}'
    elif 'N' <= hemisphere <= 'X':
        crs = f'EPSG:326{tile_code}'
    else:
        raise ValueError(f'Invalid hemisphere code: {hemisphere}')
    
    return CRS.from_string(crs)


def load_GEDI_data(path_gedi, tile_geom, tile_name) :
    """
    This function loads the GEDI footprints whose geometry intersect the provided geometry.

    Args:
    - path_gedi: string, path to the GEDI data directory.
    - tile_geom: shapely geometry, geometry of the Sentinel-2 tile (in the same CRS as the GEDI data, WGS84)

    Returns:
    - GEDI: geopandas dataframe, GEDI data.
    """

    # Load the data contained in the bounding box of the tile
    GEDI = gpd.read_file(path_gedi, engine = 'pyogrio', bbox = tile_geom.bounds, rows = None)
    # And further filter, to have only the data that intersects the tile
    GEDI = GEDI[GEDI.intersects(tile_geom)]

    # And apply necessary changes
    if not GEDI.empty: GEDI = modify_GEDI_data(GEDI)

    # And reproject the GEDI data to the Sentinel-2 tile's local CRS
    crs = get_CRS_from_S2_tilename(tile_name)
    GEDI = GEDI.to_crs(crs)

    return GEDI, crs


def reproject_gedi_data(GEDI, crs, tile_bounds) :
    """
    This function reprojects the GEDI data to the same CRS as the Sentinel-2 tile, and further filters it to have only the data
    that intersects the tile.

    Args:
    - GEDI: geopandas dataframe, GEDI data.
    - crs: string, CRS of the Sentinel-2 tile.
    - tile_bounds: tuple of floats, bounds of the Sentinel-2 tile (in the tile's CRS).

    Returns:
    - GEDI: geopandas dataframe, GEDI data.
    """

    # Reproject the GEDI data to the same CRS as the Sentinel-2 tile
    GEDI = GEDI.to_crs(crs)

   # And further filter, to have only the data that intersects the tile
    bbox = box(*tile_bounds)
    GEDI = GEDI[GEDI.intersects(bbox)]

    return GEDI


def scl_quality_criteria(scl_patch) :
    """
    This function returns a quality score for the scene classification mask of a Sentinel-2 L2A product.
    The score is between 0 and 100, and is computed as the fraction of pixels that are vegetated.
    If the central pixel is not vegetated, the score is 0, independently of the other pixels.

    Args:
    - scl_patch: 2d array of the scene classification mask.

    Returns:
    - quality_score: int between 0 and 100.
    """

    # Check if the center pixel is vegetated (i.e. not obstructed in any way)
    center_pixel = scl_patch[scl_patch.shape[0] // 2, scl_patch.shape[1] // 2]
    if center_pixel != 4 : return 0

    # Return the fraction of pixels that is vegetated
    fraction = (np.count_nonzero(scl_patch == 4) / (scl_patch.shape[0] * scl_patch.shape[1]))
    
    return int(fraction * 100)


def get_gedi_data(footprint) :
    """
    Return the GEDI footprint data. We drop the `Index`, `geometry`, and `s2_product` keys.

    Args:
    - footprint: pandas.core.frame.Pandas, GEDI footprint.

    Returns:
    - gedi_data : dictionary, with the GEDI attributes as keys, and the corresponding values as values.
    """
    
    # Turn the pandas object into a dictionary
    gedi_data = footprint._asdict()
    
    # Remove the `Index`, `geometry`, and `s2_product` keys
    del gedi_data['Index'], gedi_data['geometry'], gedi_data['s2_product']

    # Remove the lat_lowest and lon_lowest keys
    del gedi_data['lat_lowest'], gedi_data['lon_lowest']
    
    return gedi_data


def get_sentinel2_1d_attributes(s2_footprint_data, s2_prod) :
    """
    This function extracts the attributes of the Sentinel-2 L2A product at hand, and returns them.
    The attributes are: the number of days between the start of the GEDI mission and the date of
    acquisition; and identifiers for the product, namely the Processing Baseline Number (PBN), and
    the Relative Orbit Number (RON). Details of the naming convention of Sentinel-2 products can be
    found at https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention.

    Args:
    - s2_footprint_data: dictionary, with the Sentinel-2 data.
    - s2_prod: string, name of the Sentinel-2 L2A product.

    Returns:
    - s2_footprint_data: dictionary, with the Sentinel-2 data, plus the attributes.
    """

    # Parse the product name
    _, _, date, pbn, ron, _, _ = s2_prod.split('_')
    
    # Get the date of acquisition and day of year
    date_obj = dt.datetime.strptime(date, "%Y%m%dT%H%M%S")
    start = dt.datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d')

    # Save the # of days since the start of the GEDI mission
    difference = (date_obj - start).days
    s2_footprint_data['date'] = difference

    # Also save information to identify the product
    s2_footprint_data['pbn'] = int(pbn.lstrip('N'))
    s2_footprint_data['ron'] = int(ron.lstrip('R'))

    return s2_footprint_data


def get_sentinel2_patch(transform, processed_bands, footprint, patch_size, s2_prod, debug = False) :
    """
    This function extracts a patch of `patch_size` pixels from the Sentinel-2 L2A product (`processed_bands`),
    centered around the GEDI `footprint`. It then checks the quality of the patch, and if it is good enough
    (in terms of being vegetated), extracts all the bands' information. Otherwise, it returns None. Once the
    patch is extracted, it also extracts 1d attributes from the Sentinel-2 L2A product name (`s2_prod`), namely
    the number of days between the start of the GEDI mission and the date of acquisition; the Julian day of year
    (encoded with cos and sin); the tile name; and the product name.

    Args:
    - processed_bands: dictionary, with the band names as keys, and the corresponding 2d arrays as values.
    - footprint: geopandas Series, GEDI footprint.
    - patch_size: tuple of ints, size of the patch to extract.
    - s2_prod: string, name of the Sentinel-2 L2A product.

    Returns:
    - s2_footprint_data: dictionary, with 'bands' and 'metadata' as keys, and the corresponding data as values.
            or None, if the patch is not good enough.
    """

    # Initialize the data
    patch_bands, patch_data = {}, {}

    # Get the row and column corresponding to the footprint center
    pt_x, pt_y = footprint.geometry.x, footprint.geometry.y
    x, y = rowcol(transform, pt_x, pt_y)
    
    # Get the size of the window to extract
    x_offset, y_offset = (patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2

    # Check that the patch fits in the tile, otherwise skip this patch
    data = processed_bands['B02']
    if (x - x_offset < 0) or (x + x_offset + 1 > data.shape[0]) \
        or (y - y_offset < 0) or (y + y_offset + 1 > data.shape[1]) :
            return None

    # Check the SCL band to see if the patch is worth keeping
    patch = processed_bands['SCL'][x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1]
    vegetation_score = scl_quality_criteria(patch)
    if vegetation_score < 20 :
        return None
    else:
        patch_data['vegetation_score'] = vegetation_score

    # Iterate over the rest of the bands
    for band, data in processed_bands.items() :
        patch = data[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1]
        patch_bands[band] = patch
    
    # Add the 1d attributes to the result dictionary
    patch_data = get_sentinel2_1d_attributes(patch_data, s2_prod)

    # We add the stacked bands to the result dictionary
    patch_data['bands'] = np.array([patch_bands[attr] for attr in S2_attrs['bands'].keys()])

    if debug:
        # FOR DEBUGGING PURPOSES
        west, north = rs.transform.xy(transform, x - x_offset -0.5, y - y_offset-0.5)
        east, south = rs.transform.xy(transform, x + x_offset+0.5, y + y_offset+0.5)
        patch_transform = rs.transform.from_bounds(west, south, east, north, patch_size[0], patch_size[1])

    return patch_data


def crop_and_pad_arrays(data, ul_row, lr_row, ul_col, lr_col):
    """
    This function crops (and pads if necessary) the icesat data to match the shape provided by
    the upper left and lower right indices.

    Args:
    - data: 2d array, icesat data.
    - (ul_row, ul_col) : tuple of ints, indices of the icesat pixel corresponding to the upper
        left corner of the Sentinel-2 tile.
    - (lr_row, lr_col) : tuple of ints, indices of the icesat pixel corresponding to the lower
        right corner of the Sentinel-2 tile.
    
    Returns:
    - data: 2d array, cropped (and padded) icesat data.
    """

    # Get the dimensions of the arrays
    icesat_height, icesat_width = data.shape

    # If any of the slicing indices are out of bounds, pad with zeros
    if ul_row < 0 or lr_row >= icesat_height or ul_col < 0 or lr_col >= icesat_width:

        print('The Sentinel-2 tile is not contained in the icesat tile.')

        # Calculate the new shape after padding
        new_height = lr_row - ul_row + 1
        new_width = lr_col - ul_col + 1

        # Create new arrays to store the padded data
        padded_data = np.zeros((new_height, new_width), dtype = data.dtype)

        # Compute the region of interest in the new padded arrays
        start_row = max(0, -ul_row)
        end_row = min(icesat_height - ul_row, lr_row - ul_row + 1)
        start_col = max(0, -ul_col)
        end_col = min(icesat_width - ul_col, lr_col - ul_col + 1)

        # Copy the original data to the new padded arrays
        padded_data[start_row : end_row, start_col : end_col] = data[max(0, ul_row) : min(icesat_height, lr_row + 1), max(0, ul_col) : min(icesat_width, lr_col + 1)]

        # Update the variables to point to the new padded arrays
        data = padded_data

    # Otherwise, simply perform the slicing operation
    else: data = data[ul_row : lr_row + 1, ul_col : lr_col + 1]

    return data


def fill_nan_with_nearest(image, nan_mask):
    """
    This function fills the NaN values in the image with the nearest non-NaN value.

    Args:
    - image: 2d array, image with NaN values.
    - nan_mask: 2d array, mask of the NaN values in the image.

    Returns:
    - filled_image: 2d array, image with NaN values filled.
    """
    
    indices = distance_transform_edt(nan_mask, return_distances = False, return_indices = True)
    filled_image = image[tuple(indices)]
    
    return filled_image


def upsampling_with_nans(image, upsampling_shape, nan_value, order) :
    """
    This function upsamples the image to the `upsampling_shape`, and fills the NaN values with the nearest non-NaN value.

    Args:
    - image: 2d array, image to upsample.
    - upsampling_shape: tuple of ints, shape of the upsampled image.
    - nan_value: int, value to use for the NaN values.
    - order: int, order of the interpolation.

    Returns:
    - upsampled_image_with_nans: 2d array, upsampled image with NaN values filled.
    """

    # Create a mask for the non-defined values
    if np.isnan(nan_value) : nan_mask = np.isnan(image)
    else: nan_mask = (image == nan_value)

    # In the original image, fill the NaN values with the nearest non-NaN value
    non_nan_image = fill_nan_with_nearest(image, nan_mask)

    # Upsample the original image
    upsampled_image = resize(non_nan_image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)

    # Upsample the NaN mask
    upsampled_nan_mask = resize(nan_mask.astype(float), upsampling_shape) > 0.5

    # Replace the NaN values in the upsampled image with NaN
    upsampled_image_with_nans = np.where(upsampled_nan_mask, nan_value, upsampled_image)

    return upsampled_image_with_nans



def get_tile(data, s2_transform, upsampling_shape, data_source, data_attrs) :
    """
    This function extracts the data for the Sentinel-2 L2A product at hand, crops it so as to perfectly match
    the Sentinel-2 tile, resamples it to 10m resolution when necessary, and returns it.

    Args:
    - data: 
    - s2_transform: affine.Affine, transform of the Sentinel-2 L2A product.
    - upsampling_shape: tuple of ints, shape of the Sentinel-2 L2A product, at 10m resolution.
    - footprint: geopandas Series, GEDI footprint.

    Returns:
    - res: dict, with the attributes as keys and the corresponding 2d arrays as values.
    """
    if data == {} : return None

    # Get the transforms
    s2_transformer, data_transformer = AffineTransformer(s2_transform), AffineTransformer(data['transform'])    

    # Upper left corner
    ul_x, ul_y = s2_transformer.xy(0, 0)
    ul_row, ul_col = data_transformer.rowcol(ul_x, ul_y)

    # Lower right corner
    lr_x, lr_y = s2_transformer.xy(upsampling_shape[0] - 1, upsampling_shape[1] - 1)
    lr_row, lr_col = data_transformer.rowcol(lr_x, lr_y)

    # Crop the data to the same bounds, padding the data if necessary
    res = {}
    for data_attr in data_attrs.keys() :
        res[data_attr] = crop_and_pad_arrays(data[data_attr], ul_row, lr_row, ul_col, lr_col)

        if data_source == 'BM' :
            res[data_attr] = upsampling_with_nans(res[data_attr].astype(data_attrs[data_attr]), upsampling_shape, NODATAVALS[data_source],3).astype(data_attrs[data_attr])


        assert res[data_attr].shape == upsampling_shape, f'{data_source} | {data_attr} | {data[data_attr].shape} | {res[data_attr].shape} | {upsampling_shape} | {ul_row} | {lr_row} | {ul_col} | {lr_col}'

    
    return res


def get_patch(tile, footprint, transform, patch_size, data_source, data_attrs) :
    """
    This function extracts a patch of `patch_size` pixels from the `tile`, centered around
    the GEDI `footprint`.

    Args:
    - tile: dictionary, with the tile data.
    - footprint: geopandas Series, GEDI footprint.
    - transform: affine.Affine, transform of the tile.
    - patch_size: tuple of ints, size of the patch to extract.

    Returns:
    - patch_data: dictionary, with 'bands' as key, and the corresponding 3d numpy array as value.
    """

    # Initialize the data
    patch_bands, patch_data = {}, {}

    # If the tile is None, fill the patch with NODATAVALS
    if tile is None:
        print(f'No {data_source} data for this patch. Filling with NODATAVALS.')
        if data_source == 'BM' :
            for attr, dtype in data_attrs.items() :
                patch_data[attr] = np.full(patch_size, fill_value = NODATAVALS[data_source], dtype = dtype)    
    
    else:

        # Get the row and column corresponding to the footprint center
        pt_x, pt_y = footprint.geometry.x, footprint.geometry.y
        x, y = rowcol(transform, pt_x, pt_y)
        
        # Get the size of the window to extract
        x_offset, y_offset = (patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2
        # Crop the tile to the patch
        for attr, data in tile.items() :
            patch = data[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1]
            assert patch.shape == (patch_size[0], patch_size[1]), f'{data_source} Patch shape is {patch.shape}, should be {patch_size} | attr {attr} | data shape {data.shape}'
            patch_bands[attr] = patch

        if data_source  == 'BM':
            patch_data = patch_bands

    return patch_data


def radiometric_offset_values(path_s2, product) :
    """
    This function extracts the BOA_QUANTIFICATION_VALUE and BOA_ADD_OFFSET_VALUES from the
    Sentinel-2 L2A product at hand, and returns them.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product.

    Returns:
    - boa_quantification_value: float, BOA_QUANTIFICATION_VALUE.
    - boa_add_offset_values: dictionary, with the physical band names as keys, and the corresponding
        BOA_ADD_OFFSET_VALUES as values.
    """

    # There is a mismatch between the names of the physical bands in the metadata file, and the
    # names of the bands in the IMG_DATA/ folder. This dictionary defines the mapping
    bands_mapping = {'B1': 'B01', 'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07', \
                    'B8': 'B08', 'B8A': 'B8A', 'B9': 'B09', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12'}

    # Parse the XML file
    tree = ET.parse(f'{join(path_s2, product)}.SAFE/MTD_MSIL2A.xml')
    root = tree.getroot()

    # Get the BOA_QUANTIFICATION_VALUE
    for elem in root.find('.//QUANTIFICATION_VALUES_LIST') :
        if elem.tag == 'BOA_QUANTIFICATION_VALUE' :
            boa_quantification_value = float(elem.text)
        else: continue

    # Get the physical bands and their ids
    physical_bands = {elem.get('bandId'): elem.get('physicalBand') \
                      for elem in root.find('.//Spectral_Information_List')}

    # Get the BOA offset values
    boa_add_offset_values = {}
    for elem in root.find('.//BOA_ADD_OFFSET_VALUES_LIST') :
        physical_band = physical_bands[elem.get('band_id')]
        actual_band = bands_mapping[physical_band]
        boa_add_offset_values[actual_band] = int(elem.text)

    return boa_quantification_value, boa_add_offset_values


def process_S2_tile(product, path_s2) :
    """
    This function iterates over the bands of the Sentinel-2 L2A product at hand; reprojects them to
    EPSG 4326; upsamples them to 10m resolution (when needed) using cubic interpolation (nearest
    neighbor for the scene classification mask); and returns them.
    
    Args:
    - product: string, name of the Sentinel-2 L2A product.
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - processed_bands: dictionary, with the band names as keys, and the corresponding 2d arrays as values.
    """
    # Get the path to the IMG_DATA/ folder of the Sentinel-2 product
    path_to_img_data = glob.glob(join(path_s2, product + '.SAFE', 'GRANULE', '*', 'IMG_DATA'))[0]

    # Get the date and tile name from the L2A product name
    _, _, date, _, _, tname, _ = product.split('_')
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])

    # Get the BOA quantification value and BOA offsets
    if dt.date(2022, 1, 25) <= dt.date(year, month, day) :
        boa_quantification_value, boa_add_offset_values = radiometric_offset_values(path_s2, product)
        print(f'BOA quantification value: {boa_quantification_value}')
        print(f'BOA add offset values: {boa_add_offset_values}')
    # Iterate over the bands    
    processed_bands = {}
    for res, bands in S2_L2A_BANDS.items() :
        for band in bands :
            # Read the band data
            with rs.open(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.tif')) as src :
                band_data = src.read(1)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
            # Turn the band into a 2d array
            if len(band_data.shape) == 3 : band_data = band_data[0, :, :]

            # Use the 10m resolution B02 band as reference
            if res == '10m' :
                if band == 'B02' :
                    # Base the other bands' upsampling on this band's
                    upsampling_shape = band_data.shape
                    # Save the transform of this band
                    _transform = transform
            
            # Upsample the band to a 10m resolution if necessary
            else :
                # Order 1 indicates nearest interpolation, and order 3 indicates cubic interpolation                
                band_data = upsampling_with_nans(band_data, upsampling_shape, NODATAVALS['S2'], 1 if band == 'SCL' else 3).astype(S2_attrs['bands'][band])

            # Correct the radiometric offset for data post 25th January 2022
            if dt.date(2022, 1, 25) <= dt.date(year, month, day) :
                transformed_band = (band_data + boa_add_offset_values[band]) / boa_quantification_value
                # Pixels that were 0 need to stay 0 (NO_DATA value)
                transformed_band[band_data == 0] = 0
                # And pixels with a negative value are set to 0
                transformed_band[transformed_band < 0] = 0
                band_data = transformed_band

            # Store the transformed band
            assert band_data.shape == (upsampling_shape[0], upsampling_shape[1])
            processed_bands[band] = band_data

    return _transform, upsampling_shape, processed_bands, crs, bounds


def GEDI_to_S2_date(GEDI_date) :
    """
    This function converts the GEDI date of acquisition (number of days since the start of the
    mission) to a Sentinel-2 analysis compatible date, in YYYY-MM-DD format.

    Args:
    - GEDI_date: string, date of the GEDI footprint.

    Returns:
    - acq_date: string, date of the Sentinel-2 L2A product.
    """
    start_of_mission = dt.datetime.strptime('2019-04-17', '%Y-%m-%d')
    acq_date = start_of_mission + dt.timedelta(days = int(GEDI_date))
    return acq_date.strftime('%Y-%m-%d')


def match_s2_product(gedi_date, tile_name, path_s2) :
    """
    For a given Sentinel-2 tile and GEDI date of acquisition (YYYY-MM-DD), select the Sentinel-2
    L2A product that is closest in time to the GEDI footprint, but not after it.

    Args:
    - gedi_date: string, date of the GEDI footprint (format YYYY-MM-DD).
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - s2_product: string, name of the Sentinel-2 L2A product.
    """

    # Get the year, month and day of the GEDI footprint
    year, month, day = GEDI_to_S2_date(gedi_date).split('-')

    # Check if there is a match for the current month, in which case we also need to
    # check the day of acquisition
    match = glob.glob(join(path_s2, f'*MSI*_{year}{month}*_T{tile_name}*.zip'))
    if match:
        match_day = basename(match[0]).split('_')[2].split('T')[0][-2:]
        if int(match_day) <= int(day):
            return basename(match[0]).strip('.zip')
    
    # Check the matches over the past six months
    year, month = int(year), int(month)
    for _ in range(1, 6) :
        month = int(month - 2) % 12 + 1
        year = year - 1 if month == 12 else year
        match = glob.glob(join(path_s2, f'*MSI*_{year}{month:02d}*_T{tile_name}*.zip'))
        if match: return basename(match[0]).strip('.zip')


def group_GEDI_by_S2(GEDI, tile_name, path_s2) :
    """
    Since many GEDI footprints were sampled on the same day, we can group them as to
    execute the `match_s2_product` fewer times. To this end, this function lists the
    unique GEDI acquisition dates, and for each of them, finds the corresponding S2
    L2A product. It then populates the GEDI dataframe with a column for the S2 product,
    and groups the footprints by S2 product.

    Args:
    - GEDI: geopandas dataframe, GEDI data.
    - tile_name: string, name of the Sentinel-2 tile.
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - groups: pandas groupby object, GEDI data grouped by S2 product.
    """
    
    # Get all unique GEDI acquisition dates
    unique_dates = GEDI['date'].unique()
    
    # For each date, find the corresponding Sentinel-2 L2A product
    matches = {gedi_date: match_s2_product(gedi_date, tile_name, path_s2) \
                                for gedi_date in unique_dates}
    
    # Populate the GEDI dataframe with a column for the Sentinel-2 product
    GEDI['s2_product'] = GEDI['date'].map(matches)

    # Drop the footprints for which no match was found
    print(f'Dropping {GEDI.s2_product.isna().sum()}/{len(GEDI)} rows, for lack of S2 match.')
    GEDI = GEDI.dropna(subset = ['s2_product'])

    # Group the GEDI data by Sentinel-2 product
    return GEDI.groupby('s2_product')


def setup_output_files(output_path, output_fname, i, N) :
    """
    The patches extraction procedure takes place in parallel, executed by N processes. As each process creates new patches (one 
    after the other), they need to be saved somewhere. This function creates and initializes an output file per process, for them
    to write the patches to. 

    Args:
    - output_path: string, path to the output directory.
    - i: int, process split i/N.
    - N: int, total number of splits.

    Returns:
    - None
    """

    # Initialize output files for each process to write the patches to
    if output_fname == '' : fname = f'data_{i}-{N}.h5'
    else: fname = f'data_{output_fname}_{i}-{N}.h5'
    with h5py.File(join(output_path, fname), 'w') as file :
        print(f'Initializing output file for split {i}/{N}.')


def initialize_results(BM_flag) :
    """
    This function initializes the results placeholder.

    Args:
    - None

    Returns:
    - (s2_data, s1_data, gedi_data, icesat_data) : dictionaries, with the Sentinel-2/Sentinel-1/GEDI/icesat attributes as keys, and empty
        lists as values.
    """

    # Populate the GEDI results placeholder
    gedi_data = {k: [] for k in GEDI_attrs.keys()}
    
    # Populate the S2 placeholder
    s2_data = {k: [] for k in list(S2_attrs['metadata'].keys()) + ['bands']}
    
    # Populate the BM placeholder
    if BM_flag: bm_data = {k: [] for k in BM_attrs.keys()}
    else: bm_data = None

    return s2_data, gedi_data, bm_data


def update_results(s2_data, gedi_data, bm_data, s2_footprint_data, gedi_footprint_data, bm_footprint_data) :
    """
    This function updates the results placeholder with the data from the current GEDI footprint.

    Args:
    - (s2_data, s1_data, gedi_data, icesat_data) : dictionaries, placeholders for the Sentinel-2, Sentinel-1 and GEDI data.
    - (s2_footprint_data, s1_footprint_data, gedi_footprint_data, icesat_footprint_data) : dictionaries, with the Sentinel-2, Sentinel-1,
        GEDI, and icesat data to update the placeholders with.
    
    Returns:
    - (s2_data, s1_data, gedi_data, icesat_data) : dictionaries, updated placeholders for the Sentinel-2, Sentinel-1 and GEDI data.
    """
    
    # Iterate over the placeholders and the new data, and update the placeholders
    for placeholder, new_data in zip([s2_data, gedi_data, bm_data], [s2_footprint_data, gedi_footprint_data, bm_footprint_data]) :
        if placeholder is None: continue
        for attr, data in new_data.items() :
            placeholder[attr].append(data)
    return s2_data, gedi_data, bm_data


def save_results(s2_data, gedi_data, bm_data, tile_name, chunk_size, file) :
    """
    This function: 1) identifies the PID of the current process; 2) opens the corresponding output file; and 3) appends the
    results to the output file, group by group, and attribute by attribute.

    Args:
    - (s2_data, bm_data, gedi_data) : dictionaries, placeholders for the Sentinel-2, ICESat-2 and GEDI data.
    - tile_name: string, name of the Sentinel-2 tile.
    - chunk_size: int, number of patches to write to file at once.
    - file: h5py File object, opened hdf5 file.

    Returns:
    - None
    """

    # Iterate over the Sentinel-2 data
    if s2_data is not None:
        for attr, data in s2_data.items() :
            
            if attr in S2_attrs['metadata'].keys() :
                dset = file[tile_name]['Sentinel_metadata'][f'S2_{attr}']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)
            
            elif attr == 'bands' :
                # Put the bands as the last axis
                data = np.moveaxis(np.array(data), 1, -1)
                
                # Save the 'normal' bands together
                dset = file[tile_name][f'S2_bands']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = data[:, :, :, : -1]
                
                # Save the SCL band separately
                dset = file[tile_name][f'S2_SCL']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = data[:, :, :, -1]

    # Iterate over the GEDI data
    for attr, data in gedi_data.items() :
        dset = file[tile_name]['GEDI'][attr]
        dset.resize(len(dset) + chunk_size, axis = 0)
        dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)
    
    # Iterate over the BM data
    if bm_data is not None:
        for attr, data in bm_data.items() :
            dset = file[tile_name]['BM'][attr]
            dset.resize(len(dset) + chunk_size, axis = 0)
            dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)


def init_h5_group(file, tile_name, patch_size, chunk_size, BM_flag) :
    """
    For a given (opened) empty hdf5 `file`, create a group for the current `tile_name`, and initialize the datasets for the
    Sentinel-2, Sentinel-1 and GEDI data. In particular, this is the structure of the datasets:
		> (group) 32TNN
					> (dataset) S2_bands, N x 15 x 15 x 12 (uint16)
										> (attrs) order
                    > (dataset) S2_SCL, N x 15 x 15 x 1 (uint8)
					> (group) Sentinel_metadata
								> (dataset) S2_vegetation_score, N x 1
								> (dataset) S2_date, N x 1
                                > (dataset) S2_pbn, N x 1
                                > (dataset) S2_ron, N x 1
					> (group) GEDI
								> (dataset) agbd, N x 1
                                > (dataset) agbd_se, N x 1
								> ...
								> (dataset) lon_decimal, N x 1
                    > (group) BM
                            > (dataset) bm, N x 15 x 15 x 1 (uint8)
                            > (dataset) std, N x 15 x 15 x 1 (float32)
                                        
    """

    # Initialize the group for the current tile
    h5_group = file.create_group(tile_name)

    # Create the S2_bands dataset
    num_s2_bands = len(S2_attrs['bands']) - 1
    dtype = S2_attrs['bands']['B01']
    dset = h5_group.create_dataset('S2_bands', shape = (0, patch_size[0], patch_size[1], num_s2_bands), \
                            maxshape = (None, patch_size[0], patch_size[1], num_s2_bands), \
                            dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1], num_s2_bands))
    dset.attrs['order'] = list(S2_attrs['bands'].keys())[:-1]

    # Create the S2_SCL dataset
    dtype = S2_attrs['bands']['SCL']
    dset = h5_group.create_dataset('S2_SCL', shape = (0, patch_size[0], patch_size[1]), \
                            maxshape = (None, patch_size[0], patch_size[1]), \
                            dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1]))


    # Create the Sentinel_metadata group and corresponding datasets
    sentinel_metadata_group = h5_group.create_group('Sentinel_metadata')
    for attr, val in S2_attrs['metadata'].items() :
        sentinel_metadata_group.create_dataset(f'S2_{attr}', shape = (0,), maxshape = (None,), dtype = val, \
                                                compression = 'gzip', chunks = (chunk_size,))

    # Create the GEDI group
    gedi_group = h5_group.create_group('GEDI')
    for attr, dtype in GEDI_attrs.items() :
        if dtype == str: dtype = h5py.string_dtype()
        gedi_group.create_dataset(attr, shape = (0,), maxshape = (None,), dtype = dtype, compression = 'gzip', chunks = (chunk_size,))

    # Create the BM group
    if BM_flag:
        bm_group = h5_group.create_group('BM')
        for attr, dtype in BM_attrs.items() :
            bm_group.create_dataset(attr, shape = (0, patch_size[0], patch_size[1]), maxshape = (None, patch_size[0], patch_size[1]), dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1]))


def load_BM_data(path_bm, tile_name) :
    """
    This function loads the BM data for the current tile and year.

    Args:
    - path_bm: string, path to the BM data directory.
    - tile_name: string, name of the Sentinel-2 tile.
    - year: str, year of the Sentinel-2 product.

    Returns:
    - BM: dictionary, with the BM attributes as keys, and the corresponding values as values.
    """

    BM = {}

    with rs.open(join(path_bm, f'{tile_name}_cropped_mosaic.tif')) as src: #changed
        BM['bm'] = src.read(1)
        BM['std'] = src.read(2)
        BM['transform'] = src.transform

    return BM

