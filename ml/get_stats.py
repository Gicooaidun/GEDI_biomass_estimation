"""

This script computes various statistics on the patches that belong to the specified tiles.

Execution:
    python get_stats.py --path_h5 /path/to/patches
                        --fnames fname1.h5 fname2.h5 ...
                        --output_fname output_fname
                        --num_patches_sim num_patches_sim
                        --tiles_to_process tile1 tile2 ...
                        --parallel (optional)
                        --combine (optional)

"""

############################################################################################################################
# IMPORTS

import argparse
import numpy as np
from collections import defaultdict
import h5py
from os.path import join
import pickle
from copy import deepcopy

NODATAVALS = {'S2_bands' : 0, 'BM': -9999.0}

############################################################################################################################
# Helper functions

def setup_parser() :
    """ 
    Setup the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Paths arguments
    parser.add_argument('--path_h5', type = str, help = 'Path to the directory with the .h5 files.')
    parser.add_argument('--fnames', type = str, nargs = '*', help = 'Names of the files to consider.', required = True)
    parser.add_argument('--output_fname', type = str, help = 'Name of the output file.', default = '')
    parser.add_argument('--num_patches_sim', type = int, help = 'Number of patches to read simultaneously.', required = False, default = 10000)
    parser.add_argument('--tiles_to_process', type = str, nargs = '*', help = 'Names of the tiles to process.', required = False, default = [])
    parser.add_argument('--parallel', action = 'store_true', help = 'Whether to run the script in parallel.')
    parser.add_argument('--combine', action = 'store_true', help = 'Whether to combine the statistics of the files.')
    
    # Parse the arguments
    args = parser.parse_args()

    if args.parallel :
        assert len(args.fnames) == 1, 'If running in parallel, only one file can be processed at a time.'
        args.output_fname = f"_parallel_{args.fnames[0].rstrip('.h5')}"

    else:

        assert args.output_fname != '', 'Please provide an output file name.'

        if not args.output_fname.startswith('_'):
            args.output_fname = '_' + args.output_fname
    

    return args.path_h5, args.fnames, args.output_fname, args.num_patches_sim, args.tiles_to_process, args.parallel, args.combine


def get_stats(data, stats) :
    """
    This function computes the statistics of the data and updates the stats dictionary.

    Args:
    - data (np.array): The data to compute the statistics on.
    - stats (dict): The dictionary containing the statistics.

    Returns:
    - stats (dict): The updated dictionary containing the statistics.
    """

    # Check that there are no infinite or NaN values
    if np.any(np.isinf(data)) or np.any(np.isnan(data)):
        if np.any(np.isinf(data)): print('Infinite values found in the data. Skipping...')
        if np.any(np.isnan(data)): print('NaN values found in the data. Skipping...')
        exit()

    # Cast everything to float32
    data = data.astype(np.float32)

    if data.size == 0: 
        return stats

    # Calculate the statistics
    mean = np.mean(data, dtype = np.float32)
    std = np.std(data, dtype = np.float32)
    num_samples = data.size
    min_val = min(np.min(data), stats['min'])
    max_val = max(np.max(data), stats['max'])
    vals, counts = np.unique(data, return_counts = True)

    # Populate the statistics
    stats['min'] = min_val
    stats['max'] = max_val

    for stat in ['mean', 'std', 'num_samples']:
        stats[stat].append(locals()[stat])

    for val, count in zip(vals, counts):
        stats['hist'][val] += count

    return stats


def init_stats() :
    """
    This function initializes the dictionary containing the statistics.

    Returns:
    - stats (dict): The dictionary containing the statistics.
    """
    
    base_stats = {'mean': [], 'std': [], 'min': np.inf, 'max': -np.inf, 'hist' : defaultdict(int), 'num_samples': []}
    return {'S2_bands': {b: deepcopy(base_stats) for b in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']},
        'BM': {b: deepcopy(base_stats) for b in ['bm', 'std']},
        'Sentinel_metadata': {b: deepcopy(base_stats) for b in ['S2_vegetation_score', 'S2_date']},
        'GEDI': {b: deepcopy(base_stats) for b in ['agbd', 'agbd_se', 'rh98', 'date']}
    }


def composite_mean_std(means, stds, num_samples) :
    """
    This function computes the composite mean and standard deviation of the data.
    The formula can be found at https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation

    Args:
    - means (list): The list of means of the data.
    - stds (list): The list of standard deviations of the data.
    - num_samples (list): The list of number of samples of the data.

    Returns:
    - composite_mean (float): The composite mean of the data.
    - composite_std (float): The composite standard deviation of the data.
    """

    composite_mean = (1 / sum(num_samples)) * sum([mean * num_sample for mean, num_sample in zip(means, num_samples)])

    A = (1 / (sum(num_samples) - len(num_samples)))
    B = sum([(num_sample - 1) * (std ** 2) + num_sample * (mean ** 2) for mean, std, num_sample in zip(means, stds, num_samples)])
    C = sum(num_samples) * (composite_mean ** 2)

    composite_var = A * (B - C)
    composite_std = np.sqrt(composite_var)

    return composite_mean, composite_std


def get_percentiles(hist) :
    """
    This function computes the 1st and 99th percentiles of the data.

    Args:
    - hist (dict): The dictionary containing the histogram of the data.

    Returns:
    - p1 (float): The 1st percentile of the data.
    - p99 (float): The 99th percentile of the data.
    """

    values, counts = np.array(list(hist.keys())), np.array(list(hist.values()))
    frequencies = np.cumsum(counts) / np.sum(counts)

    # Get the 1st percentile
    p1 = values[len(frequencies[frequencies < 0.01])]

    # Get the 99th percentile
    p99 = values[len(frequencies[frequencies < 0.99])]

    return p1, p99


def aggregate_stats(stats, parallel = False) :
    """
    This function aggregates the statistics over all patches.

    Args:
    - stats (dict): The dictionary containing the statistics.

    Returns:
    - final_stats (dict): The dictionary containing the aggregated statistics.
    """

    final_stats = {}

    for key in stats.keys():
    
        print(key)

        final_stats[key] = {}


        for band in stats[key].keys():

            if stats[key][band]['num_samples'] == [] : 
                print('skipping')
                continue
                

            if parallel :
                final_mean, final_std = composite_mean_std(stats[key][band]['mean'], stats[key][band]['std'], stats[key][band]['num_samples'])
                final_num_samples = sum(stats[key][band]['num_samples'])
                final_min, final_max = stats[key][band]['min'], stats[key][band]['max']
                
                final_stats[key][band] = {'mean': final_mean, 'std': final_std, 'num_samples': final_num_samples,\
                                        'min': final_min, 'max': final_max, 'hist' : stats[key][band]['hist']}

            else:
                final_mean, final_std = composite_mean_std(stats[key][band]['mean'], stats[key][band]['std'], stats[key][band]['num_samples'])
                final_min, final_max = stats[key][band]['min'], stats[key][band]['max']
                p1, p99 = get_percentiles(stats[key][band]['hist'])

                final_stats[key][band] = {'mean': final_mean, 'std': final_std, 'min': final_min, 'max': final_max, 'p1': p1, 'p99': p99}
            
    return final_stats


############################################################################################################################
# Execute

if __name__ == '__main__':

    # Parse the arguments
    path_h5, fnames, output_fname, num_patches_sim, tiles_to_process, parallel, combine = setup_parser()


    if combine :
        print('Combining the statistics of the files:', fnames)

        stats = init_stats()
        
        for fname in fnames:
            with open(join(path_h5, fname), 'rb') as f:
                file_stats = pickle.load(f)
            
            for key in file_stats.keys():

                for band in file_stats[key].keys():
                    for stat in file_stats[key][band].keys():

                        if stat in ['mean', 'std', 'num_samples']:
                            stats[key][band][stat].append(file_stats[key][band][stat])
                        if stat == 'min' : stats[key][band][stat] = min(stats[key][band][stat], file_stats[key][band][stat])
                        if stat == 'max' : stats[key][band][stat] = max(stats[key][band][stat], file_stats[key][band][stat])
                        if stat == 'hist' : 
                            for val, count in file_stats[key][band][stat].items():
                                stats[key][band][stat][val] += count

    elif tiles_to_process == []:

        print(f'Processing the files: {fnames}')

        # Initialize the statistics
        stats = init_stats()

        # Iterate over the files
        for fname in fnames :
            print(f'>> Processing {fname}')
            with h5py.File(join(path_h5, fname), 'r') as f:

                # Iterate over the tiles
                total_num_tiles = len(f.keys())
                for t_num, tile in enumerate(f.keys()) :

                    if t_num % 10 == 0:
                        print(f'    Processing tile {t_num + 1}/{total_num_tiles}')

                    total_len = len(f[tile]['GEDI']['agbd'])

                    # Iterate over all the datasets in the file
                    for key in f[tile].keys():

                        match key:

                            case 'S2_bands' :
                                dataset = f[tile][key] # (num_patches, 15, 15, num_bands)
                                band_order = dataset.attrs['order']

                                # Iterate over the bands
                                for band_idx, band in enumerate(band_order): 
                                    for i in range(0, total_len, num_patches_sim):
                                        
                                        data = dataset[i : i + num_patches_sim, :, :, band_idx] # (num_patches, 15, 15)

                                        # Get the BOA flag
                                        actual_num_patches = min(data.shape[0], num_patches_sim)
                                        if 'S2_BOA' in f[tile]['Sentinel_metadata'].keys() : boa_offsets = f[tile]['Sentinel_metadata']['boa_offset'][i : i + actual_num_patches]
                                        else: boa_offsets = np.zeros(actual_num_patches)

                                        # Get the surface reflectance values
                                        sr_data = (data - boa_offsets[:, np.newaxis, np.newaxis] * 1000) / 10000
                                        sr_data[data == NODATAVALS[key]] = NODATAVALS[key]
                                        sr_data[sr_data < 0] = 0

                                        # Get the statistics
                                        data = sr_data[sr_data != NODATAVALS[key]]
                                        stats[key][band] = get_stats(data, stats[key][band])
                            
                            case 'BM':
                                for attr in f[tile][key].keys():
                                    for i in range(0, total_len, num_patches_sim):
                                        dataset = f[tile][key][attr] # (num_patches, 15, 15)
                                        data = dataset[i : i + num_patches_sim, :, :]
                                        data = data[data != NODATAVALS[key]]
                                        stats[key][attr] = get_stats(data, stats[key][attr])

                            case 'Sentinel_metadata':
                                for attr in f[tile][key].keys():
                                    if attr in ['S2_vegetation_score', 'S2_date']:
                                        for i in range(0, total_len, num_patches_sim):
                                            dataset = f[tile][key][attr] # (num_patches, 1)
                                            data = dataset[i : i + num_patches_sim]
                                            stats[key][attr] = get_stats(data, stats[key][attr])
                                    else: continue
                            
                            case 'GEDI':
                                for attr in f[tile][key].keys():
                                    if attr in ['agbd', 'agbd_se', 'rh98', 'date']:
                                        for i in range(0, total_len, num_patches_sim):
                                            dataset = f[tile][key][attr] # (num_patches, 1)
                                            data = dataset[i : i + num_patches_sim]
                                            stats[key][attr] = get_stats(data, stats[key][attr])
                                    else: continue

    else:

        print(f'Processing the files: {fnames}')
        print(f'Processing the tiles: {tiles_to_process}')

        # Initialize the statistics
        stats = init_stats()

        # Iterate over the files
        for fname in fnames :
            print(f'>> Processing {fname}')
            with h5py.File(join(path_h5, fname), 'r') as f:

                # Iterate over the tiles
                total_num_tiles = len(f.keys())
                for t_num, tile in enumerate(f.keys()) :

                    if tile not in tiles_to_process : continue
                    print(f'    Processing tile {tile}')

                    total_len = len(f[tile]['GEDI']['agbd'])

                    # Iterate over all the datasets in the file
                    for key in f[tile].keys():

                        match key:

                            case 'S2_bands' :
                                dataset = f[tile][key] # (num_patches, 15, 15, num_bands)
                                band_order = dataset.attrs['order']

                                # Iterate over the bands
                                for band_idx, band in enumerate(band_order): 
                                    for i in range(0, total_len, num_patches_sim):
                                        
                                        data = dataset[i : i + num_patches_sim, :, :, band_idx] # (num_patches, 15, 15)

                                        # Get the BOA flag
                                        actual_num_patches = min(data.shape[0], num_patches_sim)
                                        if 'S2_BOA' in f[tile]['Sentinel_metadata'].keys() : boa_offsets = f[tile]['Sentinel_metadata']['boa_offset'][i : i + actual_num_patches]
                                        else: boa_offsets = np.zeros(actual_num_patches)

                                        # Get the surface reflectance values
                                        sr_data = (data - boa_offsets[:, np.newaxis, np.newaxis] * 1000) / 10000
                                        sr_data[data == NODATAVALS[key]] = NODATAVALS[key]
                                        sr_data[sr_data < 0] = 0

                                        # Get the statistics
                                        data = sr_data[sr_data != NODATAVALS[key]]
                                        stats[key][band] = get_stats(data, stats[key][band])
                            
                            case 'BM':
                                for attr in f[tile][key].keys():
                                    for i in range(0, total_len, num_patches_sim):
                                        dataset = f[tile][key][attr] # (num_patches, 15, 15)
                                        data = dataset[i : i + num_patches_sim, :, :]
                                        data = data[data != NODATAVALS[key]]
                                        stats[key][attr] = get_stats(data, stats[key][attr])

                            case 'Sentinel_metadata':
                                for attr in f[tile][key].keys():
                                    if attr in ['S2_vegetation_score', 'S2_date']:
                                        for i in range(0, total_len, num_patches_sim):
                                            dataset = f[tile][key][attr] # (num_patches, 1)
                                            data = dataset[i : i + num_patches_sim]
                                            stats[key][attr] = get_stats(data, stats[key][attr])
                                    else: continue
                            
                            case 'GEDI':
                                for attr in f[tile][key].keys():
                                    if attr in ['agbd', 'agbd_se', 'rh98', 'date']:
                                        for i in range(0, total_len, num_patches_sim):
                                            dataset = f[tile][key][attr] # (num_patches, 1)
                                            data = dataset[i : i + num_patches_sim]
                                            stats[key][attr] = get_stats(data, stats[key][attr])
                                    else: continue

    
    # Aggregate the statistics over all patches
    try: 
        final_stats = aggregate_stats(stats, parallel)
    
    except Exception as e:
        print(f'Error: {e}')
        print('The statistics could not be aggregated. Saving the intermediary values and exiting...')
        with open(join(path_h5, f'intermediary_stats{output_fname}.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        exit(1)
    
    if not parallel :

        # Cast everything to float32
        for key in final_stats.keys():

            for band in final_stats[key].keys():
                for stat in final_stats[key][band].keys():
                    final_stats[key][band][stat] = np.float32(final_stats[key][band][stat])

    # Save the statistics in a .pkl file
    with open(join(path_h5, f'statistics{output_fname}.pkl'), 'wb') as f:
        pickle.dump(final_stats, f)