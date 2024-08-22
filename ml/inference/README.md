# Machine Learning Inference

This folder contains code to run inference once one has created a model in the parent folder.

## Files in this Folder

### 1. `creating_inference_tifs_folder.ipynb`
This notebook contains a script to convert the `.npy` inference output into `.tif` files.

### 2. `get_tilenames_above_gedi.py`
This code creates a `.txt` file which includes the tiles for which we want to run inference. In our case we were interested in a section of boreal forest for which we didn't have GEDI ground truth. This can also be done manually. 

### 3. `inference.py`
This file runs inference on the tiles for which we have no GEDI data. It creates overlapping patches which are weighted as to have less influence towards the edges, fuses these and saves the results as a `.npy` file. 

## How to Use

First create reliable train/test/val sets by calculating and analyzing the statistics and visualizing the tiles.

Then train your model of choice and run inference.

