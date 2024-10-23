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

from transforms import TransformGridImage, ToLabelOutput, ToLabelOutputFleni

def select_criteria_accuracy_aggregate(aggregate, outputs, labels):
  if aggregate == None: # init
      aggregate = 0
  _, preds = torch.max(outputs, 1)
  return aggregate + torch.sum(preds == labels.data)

def select_criteria_accuracy_epoch_result(aggregate, dataset, num_classes = 3):
  return aggregate.double() / len(dataset)

def select_criteria_f1AD_aggregate(aggregate, outputs, labels):
  _, preds = torch.max(outputs, 1)
  if aggregate == None: # init
    aggregate = {
        "preds": torch.clone(preds),
        "labels": torch.clone(labels.data),
    }
    return aggregate 
  else:
    aggregate["preds"] = torch.cat((aggregate["preds"], torch.clone(preds)))
    aggregate["labels"] = torch.cat((aggregate["labels"], torch.clone(labels.data)))
    return aggregate 

def select_criteria_f1AD_epoch_result(aggregate, dataset, num_classes = 3):
  fn = 0
  fp = 0
  tp = 0
  preds = aggregate["preds"]
  labels = aggregate["labels"]
  AD = ToLabelOutput(num_classes)("AD")
  for i in range(0, len(preds)):
    if preds[i] != AD and labels[i] != AD: # solo nos interesa AD
      continue
    if preds[i] == AD and labels[i] == AD:
      tp += 1
    if preds[i] == AD and labels[i] != AD:
      fp += 1
    if preds[i] != AD and labels[i] == AD:
      fn += 1

  # avoid division by zero
  # https://stats.stackexchange.com/questions/8025/what-are-correct-values-for-precision-and-recall-when-the-denominators-equal-0
  if tp == 0 and (fp != 0 or fn != 0):
    print("\1")
    return 0
  elif tp == 0:
    print("\2")
    return 1

  recall = 1.0 * tp / (tp + fn)
  precision = 1.0 * tp / (tp + fp)
  return 2.0 * precision * recall / (precision + recall)
