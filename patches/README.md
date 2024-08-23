# Dataset Creation

This folder contains scripts used for creating the dataset for our machine learning models.

## Files in this Repository

### `create_filenames_txt.py`
This script generates a text file (`tile_names.txt`) containing the filenames of the Sentinel-2 tiles.

### `create_patches.py`
This script is responsible for creating the dataset used by the machine learning pipeline. It groups the ICESat and Sentinel-2 data around a GEDI point and generates square patches.

### `helper_patches.py`
This script contains helper functions for `create_patches.py`.

## How to Use

To generate the dataset follow:

1. Run `create_filenames_txt.py` to generate a textfile with the names of all the Sentinel-2 tiles we want to include in the dataset.

2. Run the `create_patches.py` script to generate the patches.

    Make sure to include necessary flags:

    ```python3 create_patches.py --year 2019 --patch_size 15 15 --chunk_size 1 --output_fname "test" --BM --i 20 --N 50 --tilenames tile_names.txt```

