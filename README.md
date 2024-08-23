# GEDI Biomass Estimation Pipeline

This repository contains a machine learning pipeline designed to create models that predict GEDI biomass from ICESat and Sentinel-2 data. The project is organized into several folders, each serving a specific purpose within the pipeline:

- **data**: Contains the raw data used for model training and evaluation.
- **data_preprocessing**: Includes scripts and tools for preprocessing the raw data.
- **ml**: Houses the machine learning models and related scripts for training and evaluation.
- **patches**: Responsible for creating the dataset by generating patches from the raw data.
- **statistics**: Provides tools and scripts for analyzing the data before proceeding with model training.

For more information on the individual steps, refer to the READMEs in the folders.

### Additional notes:
In this project the variable 'bm' refers to ICESat-2 biomass.