# Machine Learning

This folder contains code to analyze train/test/val sets, train ML models on the data and then run inference.

## Files in this Folder

### `analyze_stats_sets.ipynb`
This python notebook is used to analyze the statistics saved in `calculate_all_stats.py`. It searches for the split where the mean biomass values are the closest. The first is the most relevant, outputting the best random seed, while the subsequent code can be used to further analyse the statistics. 

### `calculate_all_stats.py`
This code tries out different train/test/val splits and saves the statistics. We observed that arbitrary train/test/validation splits led to sampling bias, resulting in inconsistent data distributions and varying model performance. To mitigate this, we now save statistics for each set which can then be used to select a random seed which creates sets which exhibit the least distribution mismatch. In this code the user has to manually specify the path to the h5 files and the filenames (not as flags but in the file).

### `create_splits.ipynb`
This notebook visualizes the train/test/validation splits on a map for the user to see if the optimal split calculated above seems appropriate.

### `dataloader.py`
The dataloader used for the models. Make sure to specify the correct filenames in the main function when executing the code by itself to check if the dataloader works as expected.

### `train.py`
This script is used to train a single ML model on the dataset. It takes the preprocessed data and trains a simple CNN which is then saved for later use.

### `train_ensemble.py`
This script is used to train an ensemble of ML models on the dataset. It creates 5 different models with different weight initializations. The predictions of these models are then combined to make a final prediction. It also outputs the standard deviation of the predictions to give an idea of the range of variability in the models' results.

## How to Use

First create reliable train/test/val sets by calculating and analyzing the statistics and visualizing the tiles.

Then train your model of choice and run inference.

