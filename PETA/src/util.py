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
import random

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndi.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        
        out = ndi.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)

        if trim_top < 0 and trim_left < 0:
            outImage = np.zeros((h,w)) # New image of the original size
            outImage[-trim_top:-trim_top + h, -trim_left: -trim_left + h] = out
            out = outImage
        else:
            if trim_top < 0 or trim_left < 0:
                raise Exception("Este caso no lo contemplé y no debería pasar")
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def printFile(text, file):
  print(text)
  if file != None:
      file.write(text + "\n")


def logDebug(str, debug = False):
    if debug:
        print(str)

def getClassDescription(predictedClass, num_classes = 3):
    if num_classes == 3:
        if predictedClass == 0:
            return "CN"
        elif predictedClass == 1:
            return "AD"
        else:
            return "MCI"
    else:
        if predictedClass == 0:
            return "nonAD/MCI"
        else:
            return "AD"
    
def test_model(model,dataloaders,device, phaseKey = 'val', num_classes = 3):
    classStats = [{
        'fn': 0,
        'tn': 0,
        'tp': 0,
        'fp': 0,
        'n': 0,
    } for i in range(num_classes)]
    correctlyPredicted = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders[phaseKey]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Iteramos para chequear estadisticas
            for i, correctClass in enumerate(labels.data):
              n += 1
              predictedClass = int(preds[i].item())
              correctClass = int(correctClass.item())
              classStats[correctClass]['n'] += 1
              if correctClass == predictedClass:
                  correctlyPredicted += 1
                  classStats[correctClass]['tp'] += 1
                  for j in range(num_classes):
                      if j != correctClass:
                          classStats[j]['tn'] += 1
              else:
                  classStats[correctClass]['fn'] += 1
                  classStats[predictedClass]['fp'] += 1
                  for j in range(num_classes):
                      if j != correctClass and j != predictedClass:
                          classStats[j]['tn'] += 1
    accuracy = correctlyPredicted * 1.0 / n
    return classStats, accuracy

def printClassStats(stats, f = None):
  recall = sensitivity = stats['tp'] / (stats['tp'] + stats['fn']) # prob positive test result
  specificity = stats['tn'] / (stats['tn'] + stats['fp'])          # prob negative test result
  if stats['tp'] + stats['fp'] > 0:
    precision = stats['tp'] / (stats['tp'] + stats['fp'])          # prob of recognized positive actually correct
  else:
    precision = 1
    printFile("Setting precision as 1 but no positive value has been reported, so this is placeholder", f)
  if precision + recall == 0:
    printFile("Setting f1 as 0 because precision + recall is ZERO", f)
    f1 = 0.0
  else:
    f1 = 2 * (precision * recall) / ( precision + recall )
  printFile("Sensitivity (%): " + str(round(sensitivity * 100)), f)
  printFile("Specificity (%): " + str(round(specificity * 100)), f)
  printFile("Precision  (%): " + str(round(precision * 100)), f)
  printFile("F1 Score  (%): " + str(round(f1 * 100)), f)
  printFile("Number of images: " + str(stats['n']), f)
  return recall, specificity, precision, f1

# Returns testY and predY for a data loader
def collectAllData(dataloader, model, device, num_classes):
  testY = np.empty((0,))
  predY = np.empty((0,num_classes))
  model.eval()
  with torch.no_grad():
    for inputs, labels in dataloader:
      inputs = inputs.to(device)

      outputs = model(inputs)

      soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
      soft_outputs = soft_outputs.cpu().detach().numpy()

      testY = np.concatenate([testY, labels.detach().numpy()])
      predY = np.concatenate([predY, soft_outputs], axis = 0) 

      torch.cuda.empty_cache()

  return testY, predY
