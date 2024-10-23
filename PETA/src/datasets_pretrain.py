import os
import time
import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, SubsetRandomSampler
import torchvision
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import scipy.ndimage as ndi
import json
from util import logDebug

from datasets import ADNIDataset, BaseDataset
from transforms import ToLabelOutput, TransformReduced3DImage, Transform3DImage, ToLabelOutputConfigurable
from util import collectAllData

def get_random_subset_sampler(subset_size):
    return SubsetRandomSampler(torch.randperm(len(dataset))[:subset_size])

class PretrainDatasetBuilder(object):
    def __init__(self, dataAugmentation, dataLoaderArgs, normalization, subset_size = None, adni = True, fleni = True, adniMini = False, merida = False, chinese = False, num_classes = 2, axial_slicelen = 16, miniset_size = None, replacement = False, random_state = 22, augmentation = 'one', adniDatasetCSV = "../../Sets/Muestra3700_80_10_10_dxlast_train.csv", adniTruthLabel = "DX_last", verbose = False):
        # self.adniArgs = adniArgs
        # self.fleniArgs = fleniArgs
        # self.meridaArgs = meridaArgs
        # self.chineseArgs = chineseArgs
        # self.oasisArgs = oasisArgs
        # self.dataLoaderArgs = dataLoaderArgs
        self.adni = adni
        self.fleni = fleni
        self.adniMini = adniMini
        self.merida = merida
        self.chinese = chinese

        if axial_slicelen == 16:
            transform3D = TransformReduced3DImage(**dataAugmentation)
        else:
            transform3D = Transform3DImage(yDim = axial_slicelen, augmentation = augmentation, **dataAugmentation, verbose = verbose)

        self.axial_slicelen = axial_slicelen
        self.dataAugmentation = dataAugmentation
        self.augmentation = augmentation
        self.normalization = normalization

        print("Pretrain Dataset bilder: axial slicelen")
        print(axial_slicelen)

        self.transform3D = transform3D

        self.transform = torchvision.transforms.Compose([
            transform3D,
            torchvision.transforms.ToTensor(),
            normalization
        ])

        self.num_classes = num_classes
        self.subset_size = subset_size

        self.ADNI_GROUP_LABEL = 0
        self.FLENI_GROUP_LABEL = 1
        self.MERIDA_GROUP_LABEL = 2
        self.CHINESE_GROUP_LABEL = 3

        self.dataLoaderArgs = dataLoaderArgs
        self.miniset_size = miniset_size
        self.replacement = replacement
        self.random_state = random_state

        self.adniDatasetCSV = adniDatasetCSV

        self.adniTruthLabel = adniTruthLabel

        self.verbose = verbose
    
    def build(self):
        datasets = []

        self.class_weights = {}

        if self.adni:
            print("Pretrain Loader: Loading ADNI dataset")
            print(f"Using {self.adniDatasetCSV} as ADNI dataset")
            csv = pd.read_csv(self.adniDatasetCSV)
            csv['dataset'] = self.ADNI_GROUP_LABEL

            if self.miniset_size != None:
                csv = csv.sample(n=self.miniset_size, random_state = self.random_state)
                print(f"Reducing adni set to fixed n = {self.miniset_size} and random_state = {self.random_state}")
                print("Selected samples:")
                print(str(csv['Image Data ID']))
 
            imagesDir = "/home/eiarussi/Proyectos/Fleni/ADNI-MUESTRA-FULL-stripped-preprocessed3/"
            dataset = ADNIDataset('train', csv, imagesDir, transform = self.transform, truthLabel = self.adniTruthLabel, target_transform = ToLabelOutput(numClasses = 2), verbose = self.verbose)

            self.class_weights[self.ADNI_GROUP_LABEL] = len(dataset)
            datasets.append(dataset)

        if self.fleni:
            print("Pretrain Loader: Loading Fleni dataset")
            csvFile = "../../Sets/fleni-unclassified.csv"
            csv = pd.read_csv(csvFile)
            csv['dataset'] = self.FLENI_GROUP_LABEL

            if self.miniset_size != None:
                csv = csv.sample(n=self.miniset_size, random_state = self.random_state)
                print(f"Reducing Fleni set to fixed n = {self.miniset_size} and random_state = {self.random_state}")
                print("Selected samples:")
                print(str(csv['pet_id']))

            # Como usamos resampleZ, tenemos que quitar el collapseYDim si hubiera
            da = self.dataAugmentation.copy()
            da["collapseYDim"] = False

            if self.axial_slicelen == 16:
                transform3D = TransformReduced3DImage(**da)
            else:
                transform3D = Transform3DImage(yDim = self.axial_slicelen, augmentation = self.augmentation, **da, resampleZ = 0.610389)
                
            fleniTransform = torchvision.transforms.Compose([
                transform3D,
                torchvision.transforms.ToTensor(),
                self.normalization
            ])

            imagesDir = "/home/eiarussi/Proyectos/Fleni/fleni-stripped-preprocessed4/"
            dataset = BaseDataset('fleniUnclassified', csv, imagesDir, studyIDLabel = 'pet_id', transform = self.transform, truthLabel = 'dataset')

            self.class_weights[self.FLENI_GROUP_LABEL] = len(dataset)
            datasets.append(dataset)

        if self.merida:
            print("Pretrain Loader: Loading Merida dataset")
            tsvFile = '../../Sets/merida.tsv'
            imagesFolder = '/home/eiarussi/Proyectos/Fleni/merida-preprocessed'
            # meridaTransforms = torchvision.transforms.Compose([
            #     TransformGridImage(),
            #     torchvision.transforms.ToTensor(),
            #     torchvision.transforms.Normalize([mean, mean, mean], [std, std, std])
            # ])
            dicti = {
                "AD": 1,
                "non-AD": 0
            }
            csv = pd.read_table(tsvFile, index_col = False)
            csv['Group'] = 'non-AD'
            dataset = BaseDataset('merida', csv, imagesFolder, studyIDLabel = 'participant_id', transform = self.transform, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Group')
            self.class_weights[self.MERIDA_GROUP_LABEL] = len(dataset)
            datasets.append(dataset)
            
        if self.chinese:
            print("Pretrain Loader: Loading Chinese dataset")
            csvFile = '../../Sets/chinese.csv'
            imagesFolder = '/home/eiarussi/Proyectos/Fleni/chinese-preprocessed'
            dicti = {
                "AD": 1,
                "CN": 0
            }
            csv = pd.read_csv(csvFile)
            csv["Diagnosis"] = "CN"
            dataset = BaseDataset('chinese', csvFile, imagesFolder, studyIDLabel = 'SubjectID', transform = self.transform, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Diagnosis')
            self.class_weights[self.CHINESE_GROUP_LABEL] = len(dataset)
            datasets.append(dataset)
            
        dataset = torch.utils.data.ConcatDataset(datasets)

        # return torch.utils.data.DataLoader(dataset, **self.dataLoaderArgs, sampler=get_random_subset_sampler(self.subset_size))

        if not self.subset_size:
            print(f"Using dataset size as subset size: {len(dataset)}")
            self.subset_size = len(dataset)

        weights = []
        for d in datasets:
            dataset_len = len(d)
            dataset_sample_weight = 1. / dataset_len
            weights.append(np.repeat(dataset_sample_weight, dataset_len))
        self.weights = np.concatenate(tuple(weights))

        print("Pretrain Loader: unique weights: ")
        print(str(np.unique(self.weights)))

        # replacement = True allows picking same sample again
        sampler = torch.utils.data.WeightedRandomSampler(self.weights, num_samples = self.subset_size, replacement = self.replacement)
        dl = torch.utils.data.DataLoader(dataset, sampler=sampler, **self.dataLoaderArgs)

        return dataset, dl
