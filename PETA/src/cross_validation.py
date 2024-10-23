from __future__ import print_function, division
import os
import time
import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import scipy.ndimage as ndi
from pathlib import Path
from PIL import Image
import io
import json
import random
import sklearn.metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import sys
import json
from torchsummary import summary
from sklearn.model_selection import GroupKFold

# Devuelve K sets de train y K sets de validación para hacer cross validation
def getKFoldTrainAndValDatasets(trainDatasetCSV, valDatasetCSV, K = 1):
    if K == 1:
        return [trainCSV], [valCSV]
    
    trainCSV = pd.read_csv(trainDatasetCSV)
    valCSV = pd.read_csv(valDatasetCSV)
    concatenated = pd.concat([trainCSV, valCSV]).reset_index(drop = True)
    
    # Ensure that the concatenated DataFrame is sorted by the "Subject" column.
    concatenated.sort_values(by=["Subject"], inplace=True)

    # Extract the unique subjects from the concatenated DataFrame.
    unique_subjects = concatenated["Subject"].unique()

    print(f"{len(unique_subjects)} unique subjets found")

    # Create a GroupKFold cross-validator to ensure subject-wise splits.
    gkf = GroupKFold(n_splits=K)

    train_sets = []  # List to store training sets for each fold.
    val_sets = []    # List to store validation sets for each fold.

    groups = concatenated["Subject"]
    
    # Use enumerate to loop through each fold index and subjects.
    for fold, (train_index, val_index) in enumerate(gkf.split(X=concatenated, groups=groups)):
        train_data, val_data = concatenated.iloc[train_index], concatenated.iloc[val_index]

        # Check there is no overlap
        train_subjects = set(train_data['Subject'].unique())
        val_subjects = set(val_data['Subject'].unique())

        if train_subjects.intersection(val_subjects):
            raise Error("No debería haver overlap entre sujetos de train y de val")

        print(f"Set {fold}/{K} de train tiene {len(train_subjects)} sujetos y {len(train_index)} muestras")
        print(f"Set {fold}/{K} de val tiene {len(val_subjects)} y {len(val_index)} muestras")
        
        # Create DataFrames for the training and validation sets and store them in the lists.
        train_sets.append(train_data.copy())
        val_sets.append(val_data.copy())

    return train_sets, val_sets  # Return lists of training and validation sets for each fold.
