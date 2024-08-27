# Machine Learning Inference

This folder contains code to run inference once one has created a model in the parent folder.

## Files in this Folder

### [`creating_inference_tifs_folder.ipynb`](./creating_inference_tifs_folder.ipynb)
This notebook contains a script to convert the `.npy` inference output into `.tif` files.

### [`get_tilenames_above_gedi.py`](./get_tilenames_above_gedi.py)
This script creates a `.txt` file that includes the tiles for which we want to run inference. In our case, we were interested in a section of boreal forest for which we didn't have GEDI ground truth. This can also be done manually.

### [`inference.py`](./inference.py)
This script runs inference on the tiles for which we have no GEDI data. It creates overlapping patches that are weighted to have less influence towards the edges, fuses these, and saves the results as a `.npy` file.

## How to Use

1. First, create a text file with the names of the tiles you want to run inference on using [`get_tilenames_above_gedi.py`](./get_tilenames_above_gedi.py).
2. Then, run inference using [`inference.py`](./inference.py).
3. Finally, you can add the inference data to the ICESat data (the cropped mosaics) using [`creating_inference_tifs_folder.ipynb`](./creating_inference_tifs_folder.ipynb).
