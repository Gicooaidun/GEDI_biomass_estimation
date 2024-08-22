# Data Preprocessing

This directory contains code to preprocess data, the first step in our workflow of building a ML model to predict GEDI biomass from Sentinel-2 and ICESat data. Execute the cells of `preprocessing.ipynb`.

The subfolders (items 1.-3.) are empty before executing any code. They contain interim results (`*.tif` files) which are read by subsequent scripts.

### 1. `cropped_mosaic/`
This folder is the end result of the `preprocessing.ipynb` notebook. It contains the merged and reprojected ICESat tiles cropped to the underlying Sentinel-2 tiles.

### 2. `merged_mosaic/`
This folder stores the merged ICESat above a single S2 tile.

### 3. `reprojected_mosaic/`
This folder contains the merged tiles reprojected from the ICESat Coordinate-Reference-System into the S2 crs.

### 4. `links.txt`
A text file containing URLs for downloading data, created in one of the first few preprocessing steps.

### 5. `overlapping_tiles.csv`
A CSV file created in `preprocessing.ipynb` that lists the overlap between the ICESat and S2 tiles. It also contains additional information about the ICESat tiles.

### 6. `preprocessing.ipynb`
A Jupyter notebook that contains the code and documentation for the preprocessing steps applied to the data. This notebook provides code to download ICESat data for a given region and then creates tiles which have the extent of Sentinel-2 tiles but contain ICESat data.
