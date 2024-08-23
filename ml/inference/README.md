# Machine Learning Inference

This folder contains code to run inference once one has created a model in the parent folder.

## Files in this Folder

### `creating_inference_tifs_folder.ipynb`
This notebook contains a script to convert the `.npy` inference output into `.tif` files.

### `get_tilenames_above_gedi.py`
This code creates a `.txt` file which includes the tiles for which we want to run inference. In our case we were interested in a section of boreal forest for which we didn't have GEDI ground truth. This can also be done manually. 

### `inference.py`
This file runs inference on the tiles for which we have no GEDI data. It creates overlapping patches which are weighted as to have less influence towards the edges, fuses these and saves the results as a `.npy` file. 

## How to Use

First create a textfile with the names of the tiles you want to run inference on with `get_tilenames_above_gedi.py`

Then run inference with `inference.py` and then you can add the inference data to the icesat data (the cropped_mosaics) with `creating_inference_tifs_folder.ipynb`

