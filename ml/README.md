# Machine Learning

This folder contains code to analyze train/test/val sets, train ML models on the data, and then run inference.

## Files in this Folder

### [`analyze_stats_sets.ipynb`](./analyze_stats_sets.ipynb)
This Python notebook is used to analyze the statistics saved in `calculate_all_stats.py`. It searches for the split where the mean biomass values are the closest. The first part of the notebook is the most relevant, outputting the best random seed, while the subsequent code can be used to further analyze the statistics.

### [`calculate_all_stats.py`](./calculate_all_stats.py)
This script tries out different train/test/val splits and saves the statistics. Arbitrary train/test/validation splits often lead to sampling bias, resulting in inconsistent data distributions and varying model performance. To mitigate this, the script saves statistics for each set, which can then be used to select a random seed that creates sets with the least distribution mismatch. The user must manually specify the path to the h5 files and the filenames within the file (not as flags).

### [`create_splits.ipynb`](./create_splits.ipynb)
This notebook visualizes the train/test/validation splits on a map, allowing the user to assess whether the optimal split calculated above seems appropriate.

### [`dataloader.py`](./dataloader.py)
This script contains the dataloader used for the models. Ensure you specify the correct filenames in the main function when executing the code by itself to check if the dataloader works as expected.

### [`train.py`](./train.py)
This script is used to train a single ML model on the dataset. It takes the preprocessed data and trains a simple Convolutional Neural Network (CNN), which is then saved for later use.

### [`train_ensemble.py`](./train_ensemble.py)
This script is used to train an ensemble of ML models on the dataset. It creates 5 different models with different weight initializations. The predictions of these models are then combined to make a final prediction. The script also outputs the standard deviation of the predictions to provide an idea of the variability in the models' results.

## How to Use

1. First, create reliable train/test/val sets by calculating and analyzing the statistics using [`calculate_all_stats.py`](./calculate_all_stats.py) and [`analyze_stats_sets.ipynb`](./analyze_stats_sets.ipynb). 
2. Visualize the splits using [`create_splits.ipynb`](./create_splits.ipynb) to ensure they are appropriate.
3. Then, train your model of choice using either [`train.py`](./train.py) for a single model or [`train_ensemble.py`](./train_ensemble.py) for an ensemble of models.
4. Finally, run inference with the trained models.
