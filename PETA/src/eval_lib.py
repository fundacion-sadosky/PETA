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
import numpy as np

# Own
from util import clipped_zoom, printFile
from transforms import TransformGridImage
from datasets import FleniMyriamDataset, ADNIDataset
from select_criterias import select_criteria_accuracy_aggregate, select_criteria_accuracy_epoch_result, select_criteria_f1AD_aggregate, select_criteria_f1AD_epoch_result
from roc_auc import calculateAUCROCs, calculateAUCROCs2Classes
from util import collectAllData, printFile

def processStatsThreeClases(model_ft, device, dataloader, num_epochs, experimentOutputFolder, experimentName, executionNumber, dlName = "val", labels = [0, 1,2], labelDict = {
    0: "CN",
    1: "AD",
    2: "MCI"
  }, startEpoch = 0, endEpoch = None, printStats = False):
  bestAD = 0.0 # AUC
  bestMCI = 0.0 # AUC
  bestCN = 0.0 #AUC
  bestAcc = 0.0
  bestBalancedAcc = 0.0
  bestADEpoch = -1
  bestMCIEpoch = -1
  bestCNEpoch = -1
  bestAccEpoch = -1
  bestBalancedAccEpoch = -1
  executionNumber = 0
  if endEpoch == None:
      endEpoch = num_epochs
  for epoch in range(startEpoch, endEpoch):
    model_state_dict = torch.load(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_epoch' + str(epoch) + '.pth'), map_location=device)
    model_ft.load_state_dict(model_state_dict)

    model_ft.to(device)

    testY, predY = collectAllData(dataloader, model_ft, device, 3)
    _, preds = torch.max(torch.from_numpy(predY), 1)
    
    auc_rocs = calculateAUCROCs(testY, predY, experimentOutputFolder, experimentName, dlName, executionNumber = executionNumber, epoch = epoch, labels = labels, labelDict = labelDict)
    aucs.append(auc_rocs)

    acc = sklearn.metrics.accuracy_score(testY, preds.cpu().numpy())
    balancedAcc = sklearn.metrics.balanced_accuracy_score(testY, preds.cpu().numpy())
    accs.append(acc)
    balaccs.append(balancedAcc)

    f1 = sklearn.metrics.f1_score(testY, preds, average = None)
    recall = sklearn.metrics.recall_score(testY, preds, average = None)
    precision = sklearn.metrics.precision_score(testY, preds, average = None)

    # Epoch stats
    f = open(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_epoch' + str(epoch)  + '_' + dlName + '_stats.txt'), "w")
    
    printFile("epoch %d] CN: %.3f AD: %.3f MCI: %.3f" % (epoch, auc_rocs[0], auc_rocs[1], auc_rocs[2]), f)
    printFile("Acc: %.3f Balanced Acc: %.3f" % (acc, balancedAcc), f)
    printFile("Confusion matrix: ", f)
    printFile(str(sklearn.metrics.confusion_matrix(testY, preds.cpu().numpy())), f)
    printFile("Recall: " + str(recall), f)
    printFile("Precision: " + str(precision), f)
    printFile("F1: " + str(f1), f)
    f.close()

    # Test y preds
    # Guardamos toda la data para poder hacer análisis posteriores si fuera necesario
    f = open(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_epoch' + str(epoch)  + '_' + dlName + '_input_output.json'), "w")
    json.dump({
      "testY": str(testY), # TODO: serializar a json
      "predY": str(predY),
      "preds": str(preds)
    }, f)
    f.close()

    # Aggregating
    if auc_rocs[0] > bestCN:
      bestCN = auc_rocs[0]
      bestCNEpoch = epoch
    if auc_rocs[1] > bestAD:
      bestAD = auc_rocs[1]
      bestADEpoch = epoch
    if auc_rocs[2] > bestMCI:
      bestMCI = auc_rocs[2]
      bestMCIEpoch = epoch
    if acc > bestAcc:
      bestAcc = acc
      bestAccEpoch = epoch
    if balancedAcc > bestBalancedAcc:
      bestBalancedAcc = balancedAcc
      bestBalancedAccEpoch = epoch


  # Model aggregate stats
  f = open(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_agg_' + dlName + '_stats.txt'), "w")
  printFile("Best CN: %.3f Epoch: %d" % (bestCN, bestCNEpoch), f)
  printFile("Best AD: %.3f Epoch: %d" % (bestAD, bestADEpoch), f)
  printFile("Best MCI: %.3f Epoch: %d" % (bestMCI, bestMCIEpoch), f)
  printFile("Best Acc: %.3f Epoch: %d" % (bestAcc, bestAccEpoch), f)
  printFile("Best BalancedAcc: %.3f Epoch: %d" % (bestBalancedAcc, bestBalancedAccEpoch), f)
  if printStats:
    accs = np.array(accs)
    balaccs = np.array(balaccs)
    aucs = np.array(aucs)
    printFile("Stats:", f)
    printFile("Mean AUC: %.3f" % ( aucs.mean() ), f)
    printFile("STD AUC: %.3f" % ( aucs.std() ), f)
    printFile("Mean BALACC: %.3f" % ( balaccs.mean() ), f)
    printFile("STD BALACC: %.3f" % ( balaccs.std() ), f)
  else:
    printFile("Mean and STD not printed because startEpoch and/or endEpoch are set", f)
  f.close()


def processStatsTwoClases(model_ft, device, dataloader, num_epochs, experimentOutputFolder, experimentName, executionNumber, num_classes, dlName, labels = [0, 1,2], labelDict = {
    0: "CN",
    1: "AD",
    2: "MCI"
  }, startEpoch = 0, endEpoch = None, printStats = False):
  bestAD = 0.0 # AUC
  bestMCI = 0.0 # AUC
  bestCN = 0.0 # AUC
  bestAcc = 0.0
  bestBalancedAcc = 0.0
  bestADEpoch = -1
  bestMCIEpoch = -1
  bestCNEpoch = -1
  bestAccEpoch = -1
  bestBalancedAccEpoch = -1
  executionNumber = 0

  # For average and std
  aucs = []
  accs = []
  balaccs = []
  if endEpoch == None:
      endEpoch = num_epochs
  for epoch in range(startEpoch, endEpoch):
    model_state_dict = torch.load(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_epoch' + str(epoch) + '.pth'), map_location=device)
    model_ft.load_state_dict(model_state_dict)

    model_ft.to(device)
    
    testY, predY = collectAllData(dataloader, model_ft, device, num_classes) # acá pasamos num_clases igual
    _, preds = torch.max(torch.from_numpy(predY), 1)
    preds = preds.cpu().numpy()
    
    auc_rocs = calculateAUCROCs2Classes(testY, predY, experimentOutputFolder, experimentName, num_classes, dlName, executionNumber = executionNumber, epoch = epoch, labels = labels, labelDict = labelDict)
    aucs.append(auc_rocs)

    acc = sklearn.metrics.accuracy_score(testY, preds)
    balancedAcc = sklearn.metrics.balanced_accuracy_score(testY, preds)
    accs.append(acc)
    balaccs.append(balancedAcc)

    f1 = sklearn.metrics.f1_score(testY, preds, average = None)
    recall = sklearn.metrics.recall_score(testY, preds, average = None)
    precision = sklearn.metrics.precision_score(testY, preds, average = None)
    
    # Epoch stats
    f = open(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_epoch' + str(epoch)  + '_' + dlName + '_stats.txt'), "w")
    printFile("epoch %d] AD: %.3f" % (epoch, auc_rocs), f)
    printFile("Acc: %.3f Balanced Acc: %.3f" % (acc, balancedAcc), f)
    printFile("Confusion matrix: ", f)
    printFile(str(sklearn.metrics.confusion_matrix(testY, preds, labels = labels)), f)
    printFile("Recall: " + str(recall), f)
    printFile("Precision: " + str(precision), f)
    printFile("F1: " + str(f1), f)
    f.close()

    # Test y preds
    # Guardamos toda la data para poder hacer análisis posteriores si fuera necesario
    f = open(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_epoch' + str(epoch)  + '_' + dlName + '_input_output.json'), "w")
    json.dump({
      "testY": str(testY), # TODO: serializar a json
      "predY": str(predY),
      "preds": str(preds)
    }, f)
    f.close()

    # Aggregating
    if auc_rocs > bestAD:
      bestAD = auc_rocs
      bestADEpoch = epoch
    if acc > bestAcc:
      bestAcc = acc
      bestAccEpoch = epoch
    if balancedAcc > bestBalancedAcc:
      bestBalancedAcc = balancedAcc
      bestBalancedAccEpoch = epoch

  f = open(os.path.join(experimentOutputFolder, experimentName + '_' + str(executionNumber) + '_agg_' + dlName + '_stats.txt'), "w")
  printFile("Best AD: %.3f Epoch: %d" % (bestAD, bestADEpoch), f)
  printFile("Best Acc: %.3f Epoch: %d" % (bestAcc, bestAccEpoch), f)
  printFile("Best BalancedAcc: %.3f Epoch: %d" % (bestBalancedAcc, bestBalancedAccEpoch), f)
  if printStats:
    accs = np.array(accs)
    balaccs = np.array(balaccs)
    aucs = np.array(aucs)
    printFile("Stats:", f)
    printFile("Mean AUC: %.3f" % ( aucs.mean() ), f)
    printFile("STD AUC: %.3f" % ( aucs.std() ), f)
    printFile("Mean BALACC: %.3f" % ( balaccs.mean() ), f)
    printFile("STD BALACC: %.3f" % ( balaccs.std() ), f)
  else:
    printFile("Mean and STD not printed because startEpoch and/or endEpoch are set", f)
  f.close()
