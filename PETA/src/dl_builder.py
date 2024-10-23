import torch.optim as optim
import os
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torchvision
from torchvision import utils, models, datasets
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

import pandas as pd

import sys
sys.path.append('../../src')
from transforms import ToLabelOutput, TransformGridImage, TransformReduced3DImage, MinMaxNormalization, ToLabelOutputFleni, ToLabelOutputConfigurable
from datasets import ADNIDataset, FleniMyriamDataset, BaseDataset

class DLBuilder():
    # TODO: hacer que normalización sea un string
    def __init__(self, setsFolder, imagesRootFolder, dlArgs, normalization, gridTransform = TransformReduced3DImage(), verbose = True):
        self.setsFolder = setsFolder
        self.imagesRootFolder = imagesRootFolder
        self.dlArgs = dlArgs
        self.gridTransform = gridTransform
        self.normalization = normalization

        self.verbose = verbose
    
    def fleni100(self):
        csvFile = f"{self.setsFolder}/fleni-myriam-curated.csv"
        imagesFolder = f"{self.imagesRootFolder}/fleni-stripped-preprocessed3"
        # mean = 3383.638427734375
        # std = 7348.981426020888
        # normalization = torchvision.transforms.Normalize([mean], [std])
        valTransform = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([mean], [std])
            self.normalization
        ])
        dicti = {
            "AD": 1,
            "non-AD": 0
        }
        dataset = FleniMyriamDataset('fleni', csvFile, imagesFolder, transform = valTransform)
        return DataLoader(dataset, **self.dlArgs)

    def fleni60(self):
        csvFile = f"{self.setsFolder}/fleni-PET_clasificados60.csv"
        imagesFolder = f"{self.imagesRootFolder}/fleni-stripped-preprocessed4"
        #mean = 3364.6066073463076 # Uso el de myriam db
        #std = 7271.672596534478   # Uso el de myriam db
        # mean = 3864.730224609375
        # std = 8282.332521699427
        transforms = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([mean], [std])
            self.normalization
        ])
        dicti = {
            "AD": 1,
            "non-AD": 0
        }
        dataset = BaseDataset('fleni60', csvFile, imagesFolder, studyIDLabel = 'anon_id', transform = transforms, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Conclusion PET')
        return DataLoader(dataset, **self.dlArgs)

    def fleni600(self):
        csvFile = f"{self.setsFolder}/fleni600_limpio.csv"
        imagesFolder = f"{self.imagesRootFolder}/fleni-stripped-preprocessed4"

        transforms = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([mean], [std])
            self.normalization
        ])

        csv = pd.read_csv(csvFile)

        self.log(f"Fleni600 initial number of rows: {len(csv)}")
        
        csv = csv.drop(csv[~csv['Diagnóstico'].isin(["AD", "non-AD"])].index)

        self.log(f"Fleni600 after purguing rows with no diagnosis: {len(csv)}")
        
        dicti = {
            "AD": 1,
            "non-AD": 0
        }
        dataset = BaseDataset('fleni600', csv, imagesFolder, studyIDLabel = 'anon_id', transform = transforms, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Diagnóstico')
        return DataLoader(dataset, **self.dlArgs)

    def fleni600_train(self):
        csvFile = f"{self.setsFolder}/fleni600_80_10_10_train.csv"
        imagesFolder = f"{self.imagesRootFolder}/fleni-stripped-preprocessed4"

        transforms = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([mean], [std])
            self.normalization
        ])

        csv = pd.read_csv(csvFile)

        self.log(f"Fleni600_train initial number of rows: {len(csv)}")
        
        csv = csv.drop(csv[~csv['Diagnóstico'].isin(["AD", "non-AD"])].index)

        self.log(f"Fleni600_train after purguing rows with no diagnosis: {len(csv)}")
        
        dicti = {
            "AD": 1,
            "non-AD": 0
        }
        dataset = BaseDataset('fleni600_train', csv, imagesFolder, studyIDLabel = 'anon_id', transform = transforms, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Diagnóstico')
        return DataLoader(dataset, **self.dlArgs)

    def fleni600_val(self):
        csvFile = f"{self.setsFolder}/fleni600_80_10_10_val.csv"
        imagesFolder = f"{self.imagesRootFolder}/fleni-stripped-preprocessed4"

        transforms = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([mean], [std])
            self.normalization
        ])

        csv = pd.read_csv(csvFile)

        self.log(f"Fleni600_val initial number of rows: {len(csv)}")
        
        csv = csv.drop(csv[~csv['Diagnóstico'].isin(["AD", "non-AD"])].index)

        self.log(f"Fleni600_val after purguing rows with no diagnosis: {len(csv)}")
        
        dicti = {
            "AD": 1,
            "non-AD": 0
        }
        dataset = BaseDataset('fleni600_val', csv, imagesFolder, studyIDLabel = 'anon_id', transform = transforms, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Diagnóstico')
        return DataLoader(dataset, **self.dlArgs)

    def fleni600_test(self):
        csvFile = f"{self.setsFolder}/fleni600_80_10_10_test.csv"
        imagesFolder = f"{self.imagesRootFolder}/fleni-stripped-preprocessed4"

        transforms = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([mean], [std])
            self.normalization
        ])

        csv = pd.read_csv(csvFile)

        self.log(f"Fleni600_test initial number of rows: {len(csv)}")
        
        csv = csv.drop(csv[~csv['Diagnóstico'].isin(["AD", "non-AD"])].index)

        self.log(f"Fleni600_test after purguing rows with no diagnosis: {len(csv)}")
        
        dicti = {
            "AD": 1,
            "non-AD": 0
        }
        dataset = BaseDataset('fleni600_test', csv, imagesFolder, studyIDLabel = 'anon_id', transform = transforms, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Diagnóstico')
        return DataLoader(dataset, **self.dlArgs)
        

    def adni(self, mode, normalization, validationSet, testSet, truthLabel, numClasses = 2):
        if mode == 'validation':
            csv = f"{self.setsFolder}/{validationSet}"
        elif mode == 'test':
            csv = f"{self.setsFolder}/{testSet}"
        else:
            raise Exception(f"Invalid mode {mode}")
        imagesFolder = f"{self.imagesRootFolder}/ADNI-MUESTRA-FULL-stripped-preprocessed3"
        transforms = torchvision.transforms.Compose([
            self.gridTransform,
            torchvision.transforms.ToTensor(),
            normalization
        ])
        dataset = ADNIDataset('adni', csv, imagesFolder, transform = transforms, target_transform = ToLabelOutput(numClasses = numClasses), truthLabel = truthLabel)
        dataloader = DataLoader(dataset, **self.dlArgs)
        return dataloader


    def log(self, message):
        if self.verbose:
            print(message)
