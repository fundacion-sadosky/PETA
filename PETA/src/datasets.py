import os
import time
import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torchvision
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import scipy.ndimage as ndi
import json
from util import logDebug

class BaseDataset(Dataset):
    def __init__(self, name, csv_file, root_dir, truthLabel, studyIDLabel, transform=None, target_transform = None, cacheSize = 200, imageExtension = "nii.gz"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name = name
        if isinstance(csv_file, str):
            self.csv = pd.read_csv(csv_file)
        else:
            self.csv = csv_file # This is a pandas object already passed
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        # item_cache directamente almacena los items procesados
        self.cacheSize = cacheSize
        self.item_cache = [None] * cacheSize
        self.cachedItems = 0
        self.truthLabel = truthLabel
        self.studyIDLabel = studyIDLabel
        self.imageExtension = imageExtension

    def __len__(self):
      return int(len(self.csv))

    def itemInCache(self, idx):
        if self.cacheSize > 0 and self.item_cache[idx % self.cacheSize] != None and self.item_cache[idx % self.cacheSize]["id"] == idx:
            return self.item_cache[idx % self.cacheSize]
        else:
            return None

    def storeInCache(self, idx, image, label, metadata):
        if self.cacheSize > 0 and self.item_cache[idx % self.cacheSize] == None:  
            logDebug(self.name + "] Storing item in cache: " + str(idx))
            self.cachedItems += 1
            # Storing item in cache
            self.item_cache[idx % self.cacheSize] = {
                "id": idx,
                "label": label,
                "image": image,
                "metadata": metadata
            }
            logDebug(self.name + "] Cached items: " + str(self.cachedItems))

    def loadImage(self, studyID):
      imageFile = os.path.join(self.root_dir, studyID, f"resampled-normalized.{self.imageExtension}")
      metadataFile = os.path.join(self.root_dir, studyID, "metadata.json")
      f = open(metadataFile, "r")
      metadata = json.load(f)
      f.close()

      return nib.load(imageFile), metadata

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.itemInCache(idx):
            item = self.itemInCache(idx)
            image = item["image"]
            metadata = item["metadata"]
            label = item["label"]
        else:
            label = self.csv.iloc[idx][self.truthLabel]
            studyID = self.csv.iloc[idx][self.studyIDLabel]
          
            image, metadata = self.loadImage(studyID)
            self.storeInCache(idx, image, label, metadata)
        

        if self.transform:
            image = self.transform([image, metadata])
            
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
class FleniMyriamDataset(Dataset):
    """Fleni Myriam dataset."""

    def __init__(self, name, csv_file, root_dir, transform=None, target_transform = None, 
                 cacheSize = 200):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name = name
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        # item_cache directamente almacena los items procesados
        self.cacheSize = cacheSize
        self.item_cache = [None] * cacheSize
        self.cachedItems = 0

    def __len__(self):
      return int(len(self.csv))

    def storeInCache(self, idx, image, label, metadata):
        if self.cacheSize > 0 and self.item_cache[idx % self.cacheSize] == None:  
            logDebug(self.name + "] Storing item in cache: " + str(idx))
            self.cachedItems += 1
            # Storing item in cache
            self.item_cache[idx % self.cacheSize] = {
                "id": idx,
                "label": label,
                "image": image,
                "metadata": metadata
            }
            logDebug(self.name + "] Cached items: " + str(self.cachedItems))

    def itemInCache(self, idx):
        if self.cacheSize > 0 and self.item_cache[idx % self.cacheSize] != None and self.item_cache[idx % self.cacheSize]["id"] == idx:
            return self.item_cache[idx % self.cacheSize]
        else:
            return None

    def loadImage(self, studyID):
      imageFile = os.path.join(self.root_dir, studyID, "resampled-normalized.nii")
      metadataFile = os.path.join(self.root_dir, studyID, "metadata.json")
      f = open(metadataFile, "r")
      metadata = json.load(f)
      f.close()

      return nib.load(imageFile), metadata

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.itemInCache(idx):
            item = self.itemInCache(idx)
            image = item["image"]
            metadata = item["metadata"]
            label = item["label"]
        else:
          studyID = self.csv.iloc[idx]['pet_id']
          #subjectID = self.csv.iloc[idx, 1]
          #processFormat = self.csv.iloc[idx, 7]
          #date = self.csv.iloc[idx, 9]
          diag = self.csv.iloc[idx]['diag']
          label = diag

          image, metadata = self.loadImage(studyID)

          self.storeInCache(idx, image, label, metadata)
        
        if self.transform:
            image = self.transform([image, metadata])
            
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class ADNIDataset(Dataset):
    """ADNI dataset."""

    def __init__(self, name, csv_file, root_dir, transform=None, target_transform = None, 
                 cacheSize = 200, indexOffset = 1, truthLabel = None, verbose = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name = name
        if isinstance(csv_file, str):
            self.csv = pd.read_csv(csv_file)
        else:
            self.csv = csv_file # This is a pandas object already passed
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        # item_cache directamente almacena los items procesados
        self.cacheSize = cacheSize
        self.item_cache = [None] * cacheSize
        self.cachedItems = 0
        self.indexOffset = indexOffset
        if not truthLabel:
            raise Exception("truthLabel should be specified")
        self.truthLabel = truthLabel

        self.verbose = False

    def __len__(self):
        return int(len(self.csv))

    def storeInCache(self, idx, image, label, metadata):
        if self.cacheSize > 0 and self.item_cache[idx % self.cacheSize] == None:  
            logDebug(self.name + "] Storing item in cache: " + str(idx))
            self.cachedItems += 1
            # Storing item in cache
            self.item_cache[idx % self.cacheSize] = {
                "id": idx,
                "label": label,
                "image": image,
                "metadata": metadata
            }
            logDebug(self.name + "] Cached items: " + str(self.cachedItems))

    def itemInCache(self, idx):
        if self.cacheSize > 0 and self.item_cache[idx % self.cacheSize] != None and self.item_cache[idx % self.cacheSize]["id"] == idx:
            return self.item_cache[idx % self.cacheSize]
        else:
            return None

    def loadImage(self, studyID):
      imageFile = os.path.join(self.root_dir, studyID, "resampled-normalized.nii.gz")
      metadataFile = os.path.join(self.root_dir, studyID, "metadata.json")
      f = open(metadataFile, "r")
      metadata = json.load(f)
      f.close()

      return nib.load(imageFile), metadata

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.itemInCache(idx):
            item = self.itemInCache(idx)
            image = item["image"]
            metadata = item["metadata"]
            label = item["label"]
        else:
          # +1 por el índice que guarda Pandas
          studyID = self.csv.iloc[idx, 0 + self.indexOffset]
          label = self.csv.iloc[idx][self.truthLabel]

          image, metadata = self.loadImage(studyID)
          self.storeInCache(idx, image, label, metadata)

        if self.verbose:
            print(f"Selecting item with StudyID: {studyID}")

        if self.transform:
            image = self.transform([image, metadata])
            
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
