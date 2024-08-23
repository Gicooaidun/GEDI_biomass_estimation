from get_stats import *
import h5py
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

def process(tiles_to_process, output_fname):  
    print("creating file " + output_fname)
    # Initialize the statistics
    stats = init_stats()

    # Iterate over the files
    for fname in fnames :
        # print(f'>> Processing {fname}')
        with h5py.File(join(path_h5, fname), 'r') as f:

            # Iterate over the tiles
            total_num_tiles = len(f.keys())
            for t_num, tile in enumerate(f.keys()) :

                if tile not in tiles_to_process : continue
                # print(f'    Processing tile {tile}')

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
    with open(join(path_h5, f'balanced_distribution/statistics{output_fname}.pkl'), 'wb') as f:
        pickle.dump(final_stats, f)
    return

if __name__ == '__main__':
    # Path to the h5 files created in the create_patches.py script
    path_h5 = 'dataset/'
    num_patches_sim = 1000
    parallel = False
    combine = False
    
    train_all = []
    val_all = []
    test_all = []
    for i in range(95,10000,1):
        print("=================")
        print(f'Iteration {i}')
        # Initialize an empty dictionary to store the data
        data = {'train': [], 'val': [], 'test': []} 

        all_tiles = []
        # Iterate over all the h5 files
        for fname in os.listdir(path_h5):
            if fname.endswith('.h5'):
                with h5py.File(os.path.join(path_h5, fname), 'r') as f:
                    # Get the list of all tiles in the file
                    all_tiles.extend(list(f.keys()))

        train_tiles, test_and_val_tiles = train_test_split(all_tiles, test_size=0.35, random_state=i)
        val_tile, test_tile = train_test_split(test_and_val_tiles, test_size=0.6, random_state=i)
        data['val'].extend(val_tile)
        data['test'].extend(test_tile)
        data['train'].extend(train_tiles)

        if data['train'] in train_all or data['val'] in val_all or data['test'] in test_all:
            continue
        train_all.append(data['train'])
        val_all.append(data['val'])
        test_all.append(data['test'])

        ####################################################
        # the filenames of the h5 files in the ml/dataset/ folder
        fnames = ['data_0-5.h5', 'data_1-5.h5', 'data_2-5.h5', 'data_3-5.h5', 'data_4-5.h5']
        process(tiles_to_process=data['train'], output_fname=f'_{str(i)}_train')
        process(tiles_to_process=data['val'], output_fname=f'_{str(i)}_val')
        process(tiles_to_process=data['test'], output_fname=f'_{str(i)}_test')

