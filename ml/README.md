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

## Results
We tried different variations of a Fully Convolutional Network (defined in the [`train.py`](./train.py) file). In this WandB project the runs are logged: [code-ml-better_logging](https://wandb.ai/dose/code-ml-better_logging?nw=nwuserdsenti).

### Run Naming Conventions

The naming format of each run shows changes in the model's setup and training configuration. Each part of the name corresponds to a hyperparameter:

- **balanced**: Indicates that the train, test, and validation sets were constructed to have minimal differences in their means (i.e., balanced datasets).
- **baseline/icesat**:  
  - `baseline`: The model was trained without using ICESat data.  
  - `icesat`: The model was trained using ICESat data.
- **?Epochs**: Specifies the exact number of epochs the model was trained for. For example, `100Epochs` means the model was trained for 100 epochs.
- **LR**: Refers to the learning rate used for training. For example, `LR0.001` indicates a learning rate of 0.001.
- **Adam**: Specifies that the Adam optimizer was used during training.
- **seed24**: Indicates the random seed used to initialize the model for reproducibility. For example, `seed24` means the seed used was 24.

Additionally we tried out different channel dimensions during training. These correspond to the channel_dims argument in the definition of SimpleFCN (line 73 in [`train.py`](./train.py) ). The following table shows the channel_dims value for the different runs:

| Run Name               | Channel Dimensions                                                    |
|------------------------|-----------------------------------------------------------------------|
| **balanced_baseline**   | (16, 32, 64, 128, 64, 32, 16)                                         |
| **balanced_icesat**     | (16, 32, 64, 128, 64, 32, 16)                                         |
| **balanced_icesat_bigger** | (16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16)                   |
| **balanced_icesat_half**   | (16, 32, 64, 128)                                                   |
| **balanced_icesat_huge**   | (16, 32, 64, 128, 256, 512, 1024, 256, 128, 64, 32, 16)             |

