# Data Preprocessing

This directory contains code to preprocess data, the first step in our workflow of building a ML model to predict GEDI biomass from Sentinel-2 and ICESat data. Execute the cells of [`preprocessing.ipynb`](preprocessing.ipynb).

The subfolders are empty before executing any code. They contain interim results (`*.tif` files) which are read by subsequent scripts.

- [`cropped_mosaic/`](cropped_mosaic)
This folder is the end result of the [`preprocessing.ipynb`](preprocessing.ipynb) notebook. It contains the merged and reprojected ICESat tiles cropped to the underlying Sentinel-2 tiles.

- [`merged_mosaic/`](data_preprocessing/merged_mosaic)
This folder stores the merged ICESat above a single S2 tile.

- [ `reprojected_mosaic/`](data_preprocessing/reprojected_mosaic)
This folder contains the merged tiles reprojected from the ICESat Coordinate-Reference-System into the S2 crs.

- [ `links.txt`](data_preprocessing/links.txt)
A text file containing URLs for downloading data, created in one of the first few preprocessing steps.

- [ `overlapping_tiles.csv`](data_preprocessing/overlapping_tiles.csv)
A CSV file created in [`preprocessing.ipynb`](preprocessing.ipynb) that lists the overlap between the ICESat and S2 tiles. It also contains additional information about the ICESat tiles.

- [ `preprocessing.ipynb`](data_preprocessing/preprocessing.ipynb)
A Jupyter notebook that contains the code and documentation for the preprocessing steps applied to the data. This notebook provides code to download ICESat data for a given region and then creates tiles which have the extent of Sentinel-2 tiles but contain ICESat data.
