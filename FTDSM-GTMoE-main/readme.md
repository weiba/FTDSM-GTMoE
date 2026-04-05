# Frequency-Time Dual-Stream Mamba with Graph Topology-guided Mixture of Experts (FTDSM-GTMoE)

Source code and data for "Frequency-Time Dual-Stream Mamba with Graph Topology-Guided Mixture of Experts for Early Alzheimer's Disease Diagnosis Using rs-fMRI Time Series"



## Introduction

**FTDSM-GTMoE:** A novel framework combining Frequency-Time Dual-Stream Mamba and Graph Topology-guided Mixture of Experts for precise early AD diagnosis, mining fine-grained dynamic features to automatically localize high-risk brain regions and temporal patches.![](D:\desktop\论文3\框架.png)

## Requirements

You can install the required dependencies by running:

```python
pip install -r requirements.txt
```

In addition, CUDA 12.1 have been used on NVIDIA GeForce RTX 3090.



## File Description

### Main Code Files

- **model.py**: Contains the core implementation of the FTDSM-GTMoE model. 
- **train.py**: This script is responsible for training, validating, and testing the model. The model is trained on different fMRI classification tasks and evaluated based on the validation and test data. The best model based on validation performance is saved during the process.
- **utils.py**: This file contains utility functions used throughout the project. These include functions for data loading, calculating evaluation metrics (e.g., accuracy, sensitivity, specificity, AUC), computing adjacency matrices, normalizing adjacency matrices, and other preprocessing tasks for the fMRI data.
- **idp-test.py**: This script is used for independent testing on two tasks. It leverages AD, EMCI, and LMCI data from the ADNI3, ADNIDOD, and ADNI-GO datasets as training sets for five-fold cross-validation. Corresponding data from ADNI2 is used as the independent test set .

## Data

Due to privacy and ethical considerations and ADNI's data usage agreement, users need to apply at ADNI and download these data(https://adni.loni.usc.edu/). We provide the preprocessed fMRI data in the `data/` directory. To comply with GitHub's file size limits, each dataset has been compressed using WinRAR. Before running the training or testing scripts, please ensure you have extracted the `.rar` files into the `data/` folder so that the `.npz` files are directly accessible.

The data files include:

- **AD Diagnosis (NC vs. AD):** `AD_NC_new.rar` (contains `AD_NC_new.npz`) 
- **Early MCI Diagnosis (NC vs. LMCI):** `LMCI_NC_new.rar` (contains `LMCI_NC_new.npz`) 
- **AD and Early Subtype Classification (LMCI vs. AD):** `LMCI_AD_new.rar` (contains `LMCI_AD_new.npz`) 
- **Two MCI Subtype Classification (EMCI vs. LMCI):** `EMCI_LMCI_new.rar` (contains `EMCI_LMCI_new.npz`)

Additionally, we also provide data for independent testing located in the `data/Independent_data_center` directory. The dataset can be loaded using the `load_patch` function provided in the `utils.py` file.

### fMRI preprocessing:

DPARSF(http://rfmri.org/DPARSF)

## Usage

### Training for Classification Tasks
You can run the model for classification tasks with the following command. This example shows how to train the model for the AD vs NC classification task:

```bash
python3 train.py --data-path data/AD_NC_new.npz --batch-size 16 --hidden-size 90 --embed-size 64 --lr 1e-4 --layer-num 1
```

**Arguments for train.py:**

- `--data-path`: Path to the dataset (must be in `.npz` format).

- `--batch-size`: Batch size for training. Default is 16.

- `--hidden-size`: Size of the hidden layers. Default is 90.

- `--embed-size`: Size of the node embeddings in hypervariate brain network. Default is 64.

- `--lr`: Learning rate for the optimizer. Default is 0.001.

- `--layer-num`: Number of layers in FourierGNN. Default is 1.

  ### Independent Data Center Testing

  For independent data center testing, use the following command to evaluate the model on a separate testing dataset. (Note: The training and testing data paths are configured internally within the script for this experiment).

  Bash

  ```
  python3 independent-test.py --batch-size 16 --hidden-size 90 --embed-size 64 --lr 1e-4 --layer-num 1
  ```

  **Arguments for independent-test.py:**

  - `--batch-size`: Batch size for testing. Default is 16.
  - `--hidden-size`: Size of the hidden layers. Default is 90.
  - `--embed-size`:Size of the node embeddings in hypervariate brain network. Default is 64.
  - `--lr`: Learning rate for the optimizer. Default is 0.001.
  - `--layer-num`:  Number of layers in FourierGNN. Default is 1.

## Contact

If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me ([weipeng1980@gmail.com](mailto:weipeng1980@gmail.com)).

