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

from util import clipped_zoom, printFile, logDebug, test_model, printClassStats
from transforms import TransformGridImage, ToLabelOutput, ToLabelOutputFleni, MinMaxNormalization
from datasets import FleniMyriamDataset, ADNIDataset
from select_criterias import select_criteria_accuracy_aggregate, select_criteria_accuracy_epoch_result, select_criteria_f1AD_aggregate, select_criteria_f1AD_epoch_result
from config import loadConfig
from cross_validation import getKFoldTrainAndValDatasets

"""# Modelo"""


def train_model(model, device, dataloaders, criterion, optimizer, experimentExecutionName, experimentOutputFolder,num_epochs=25, is_inception=True, logFile = None, selection_criteria = "accuracy", save_all_epochs = False, auxEnabled = True, num_classes = 3):
    f = None
    if logFile != None:
        f = open(logFile, "w")

    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_selection_criteria_history = []
    val_selection_criteria_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_selection_criteria = 0.0

    # This allows to select different select criteria for selecting the "best epoch" one
    if selection_criteria == "accuracy":
      select_criteria_aggregate = select_criteria_accuracy_aggregate
      select_criteria_epoch_result = select_criteria_accuracy_epoch_result
    elif selection_criteria == "f1AD":
      select_criteria_aggregate = select_criteria_f1AD_aggregate
      select_criteria_epoch_result = select_criteria_f1AD_epoch_result      
    else:
      raise Exception("Invalid selection criteria")

    for epoch in range(num_epochs):
        printFile('Epoch {}/{}'.format(epoch, num_epochs - 1), f)
        printFile('-' * 10, f)

        # Each epoch has a training and validation phase
        # , 'valFleni'
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            selection_criteria_aggregate = None
            it = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        if auxEnabled: 
                          loss1 = criterion(outputs, labels)
                          loss2 = criterion(aux_outputs, labels)
                          loss = loss1 + 0.4*loss2                     
                        else:
                          loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += + torch.sum(preds == labels.data)
                # if phase == 'val':
                selection_criteria_aggregate = select_criteria_aggregate(selection_criteria_aggregate, outputs, labels.data)

                logDebug("Iteration " + str(it))
                it += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # if phase == 'val':
            epoch_selection_criteria = select_criteria_epoch_result(selection_criteria_aggregate, dataloaders[phase].dataset, num_classes)

            printFile('{} Loss: {:.4f} Acc: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, epoch_acc, selection_criteria, epoch_selection_criteria), f)

            # deep copy the model
            if phase == 'val' and epoch_selection_criteria > best_selection_criteria:
                best_selection_criteria = epoch_selection_criteria
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_selection_criteria_history.append(epoch_selection_criteria)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_selection_criteria_history.append(epoch_selection_criteria)

                if save_all_epochs:
                  fileName = os.path.join(experimentOutputFolder, experimentExecutionName + '_epoch' + str(epoch) + '.pth')
                  torch.save(model.state_dict(), fileName)
            

    time_elapsed = time.time() - since
    printFile('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), f)
    printFile('Best val {}: {:4f}'.format(selection_criteria, best_selection_criteria), f)

    # load best model weights
    model.load_state_dict(best_model_wts)
    if logFile != None:
        f.close()
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history, val_selection_criteria_history, train_selection_criteria_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

"""# Initialize and reshape inception"""

def initialize_model(model_name, num_classes, feature_extract, dropoutRate, auxEnabled, use_pretrained=True, useGradCAM = False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained, 
                                       aux_logits = True)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        print("num featurs" + str(num_ftrs))
        # Fuente: https://github.com/bdrad/petdementiapub/blob/master/petdementia_source.py
        model_ft.dropout = nn.Dropout(dropoutRate)
        model_ft.fc = nn.Sequential(
          nn.Linear(num_ftrs,1024),
          nn.ReLU(),
          nn.Linear(1024,num_classes),
        )

        if auxEnabled :
          model_ft.AuxLogits.fc = nn.Sequential(
            nn.Linear(768,num_classes), # elegido arbitrariamentoe
          )
          
        input_size = 512 

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


#"""Run"""
def run(config):
    imagesFolder = config['imagesFolder']
    fleni100ImagesFolder = config['fleni100ImagesFolder']
    fleni600ImagesFolder = config['fleni600ImagesFolder']
    trainDatasetCSV = config['trainDatasetCSV']
    valDatasetCSV = config['valDatasetCSV']
    fleni100ValDatasetCSV = config['fleni100ValDatasetCSV']
    experimentName = config['experimentName']
    experimentOutputFolder = config['experimentOutputFolder']
    experimentDescription = config['experimentDescription']
    executions = config['executions']
    model_name = config['model_name'] # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    num_classes = config['num_classes'] # Number of classes in the dataset
    batch_size = config['batch_size'] # Batch size for training (change depending on how much memory you have)
    dl_num_workers = config['dl_num_workers']
    num_epochs = config['num_epochs'] # Number of epochs to train for
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = config['feature_extract']
    usePretrained = config['usePretrained']
    auxEnabled = config['auxEnabled'] # Habilita la salida auxiliar
    learningRate = config['learningRate']
    dropoutRate = config['dropoutRate']
    normalization = config['normalization']
    trainMean = config['trainMean']
    trainStd = config['trainStd']
    deviceName = config['deviceName']
    dataAugmentation = config['dataAugmentation']
    selectCriteria = config['selectCriteria']
    validationCacheSize = config['validationCacheSize']
    trainCacheSize = config['trainCacheSize']
    calculateAUCROC = config['calculateAUCROC']
    debug = config['debug']
    doTrain = config['doTrain']
    selectCriteriaAbbrv = config['selectCriteriaAbbrv']
    truthLabel = config['truthLabel']
    crossValidationK = config['crossValidationK']

    try:
        labels = config['labels']
    except Exception: # back compat
        labels = ['CN', 'AD', 'MCI']

    #trainMean = 0.1716601789041244 #preproc3, < 0s eliminados
    #trainStd = 0.3936839672084841 #preproc3
    #trainMean = 0.1534203209139499  #preproc4, sin eliminar < 0s
    #trainStd =  0.4048895150096513   #preproc4
    if normalization == "z-score":
        zScoreNormalization = {
            #"trainMeans": [0.485, 0.456, 0.406], # ImageNet
            #"trainStds": [0.229, 0.224, 0.225].  # ImageNet
            "trainMeans": [trainMean, trainMean, trainMean],
            "trainStds": [trainStd, trainStd, trainStd]
        }

        normalizationTransform = transforms.Normalize(zScoreNormalization["trainMeans"], zScoreNormalization["trainStds"])
    elif normalization == "min-max":

        normalizationTransform = MinMaxNormalization(0, 1)
    else:
        raise Exception(f"Unknown normalization ${normalization}")
        

    # Data augmentation
    # dataAugmentation = {
    #     "angle": 8,
    #     "shiftX": 10,
    #     "shiftY": 10,
    #     "zoom": 0.1,
    #     "shear": np.pi / 16,
    # }
    # dataAugmentation = {}

    # End Config

    """# Utilidades"""

    selectCriteriaAbbrv = selectCriteriaAbbrv[selectCriteria]

    def logDebug(str):
        if debug:
            print(str)

    if deviceName.startswith("cuda") and not torch.cuda.is_available():
        raise Exception("CUDA is not available")

    if not os.path.exists(experimentOutputFolder):
      print("Creando carpeta " + experimentOutputFolder)
      os.mkdir(experimentOutputFolder)

    f = open(os.path.join(experimentOutputFolder, experimentName + "_params.txt"), "w")
    f.write("batch_size: " + str(batch_size) + "\n")
    f.write("dl_num_workers: " + str(dl_num_workers) + "\n")
    f.write("epochs: " + str(num_epochs) + "\n")
    f.write("feature_extract: " + str(feature_extract) + "\n")
    f.write("usePretrained: " + str(usePretrained) + "\n")
    f.write("auxEnabled: " + str(auxEnabled) + "\n")
    f.write("learningRate: " + str(learningRate) + "\n")
    f.write("dropoutRate: " + str(dropoutRate) + "\n")
    f.write("dataAugmentation: " + str(json.dumps(dataAugmentation)) + "\n")
    f.write("selectCriteria: " + str(selectCriteria) + "\n")
    f.write("executions: " + str(executions) + "\n")
    f.write("normalization: " + str(json.dumps(normalization)) + "\n")
    f.write("trainMean: " + str(json.dumps(trainMean)) + "\n")
    f.write("trainStd: " + str(json.dumps(trainStd)) + "\n")
    f.write("deviceName: " + str(deviceName) + "\n")
    f.write("validationCacheSize: " + str(validationCacheSize) + "\n")
    f.write("trainCacheSize: " + str(trainCacheSize) + "\n")
    f.write("calculateAURROC: " + str(calculateAUCROC) + "\n")
    f.write("truthLabel: " + str(truthLabel) + "\n")
    f.write("labels: " + str(labels) + "\n")
    f.close()

    # Copy config file for reference
    out_file = open(os.path.join(experimentOutputFolder, experimentName + 'config.json'), "w")
    json.dump(config, out_file, indent = 6)
    out_file.close()

    f = open(os.path.join(experimentOutputFolder, experimentName + "_descripcion.txt"), "w")
    f.write(experimentDescription)
    f.close()

    # Data augmentation and normalization for training
    # Just normalization for validation

    valGridArgs = {}

    data_transforms = {
        'train': transforms.Compose([
            TransformGridImage(**dataAugmentation),
            transforms.ToTensor(),
            normalizationTransform
        ]),
        'val': transforms.Compose([
            TransformGridImage(),
            transforms.ToTensor(),
            normalizationTransform
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    if crossValidationK != None:
        print(f"Usando cross validation con K = {crossValidationK}")
        train_sets, val_sets = getKFoldTrainAndValDatasets(trainDatasetCSV, valDatasetCSV, crossValidationK)
    else:
        print(f"Usando hold out validation")
        train_sets = pd.read_csv(trainDatasetCSV)
        val_sets = pd.read_csv(valDatasetCSV)

    # Ciclo de cross validation
    for k in range(0, len(train_sets)):
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, dropoutRate, auxEnabled, use_pretrained=usePretrained)

        # summary(model_ft, (3, 512, 512))

        # Print the model we just instantiated
        print(model_ft)
        
        if crossValidationK:
            print(f"Entrenando el fold set {k}{crossValidationK}")
        else:
            print("Entrenando una única vez pues no es kfold validation")
        trainDatasetCSV = train_sets[k]
        valDatasetCSV = val_sets[k]

        ads = len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'AD'])
        mci = len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'MCI']) + len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'LMCI']) + len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'EMCI'])
        cns = len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'CN'])
                
        if num_classes == 3:
            crossEntrophyWeigths = torch.tensor([cns, ads, mci], dtype=torch.float32) # Órden: CN, AD, MCI
        else:
            crossEntrophyWeigths = torch.tensor([cns + mci, ads], dtype=torch.float32) # Órden: CN/MCI, AD

        print(f"Cross entrophy weights: {crossEntrophyWeigths}")
    
        image_datasets = {
            'train': ADNIDataset('trainDL', trainDatasetCSV, imagesFolder, transform = data_transforms['train'], target_transform =ToLabelOutput(num_classes), cacheSize = trainCacheSize, truthLabel = truthLabel),
            'val': ADNIDataset('valDL', valDatasetCSV, imagesFolder, transform = data_transforms['val'], target_transform =ToLabelOutput(num_classes), cacheSize = validationCacheSize, truthLabel = truthLabel ),
            'valFleni': FleniMyriamDataset('valFleniDL', fleni100ValDatasetCSV, fleni100ImagesFolder, transform = data_transforms['val'], target_transform =ToLabelOutputFleni(num_classes), cacheSize = validationCacheSize ),
        }

        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=dl_num_workers) for x in ['train', 'val', 'valFleni']}

        # loguear cantidad de los datasets

        # fin de la parte de modificación por k fold

        # Detect if we have a GPU available
        device = torch.device(deviceName if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=learningRate)

        if crossValidationK:
            experimentNameOutput = experimentName + "_kFold" + str(k)
        else:
            experimentNameOutput = experimentName
        
        if doTrain:
          trainAndPlot(model_ft, experimentNameOutput, crossEntrophyWeigths, device, dataloaders_dict, optimizer_ft, num_epochs, model_name, selectCriteria = selectCriteria, calculateAUCROC = calculateAUCROC, auxEnabled = auxEnabled, num_classes = num_classes, experimentOutputFolder = experimentOutputFolder, executions = executions, feature_extract = feature_extract, learningRate = learningRate, batch_size = batch_size)

def trainAndPlot(model, experimentName, crossEntrophyWeigths, device, dataloaders_dict, optimizer_ft, num_epochs, model_name, selectCriteria, calculateAUCROC, auxEnabled, num_classes, experimentOutputFolder, executions, feature_extract, learningRate, batch_size):
    accuracyValues = []
    adStatValues = []
    cnStatValues = []
    mciStatValues = []

    for i in range(0, executions):
      experimentExecutionName = experimentName + '_' + str(i)
      print("--- Execution " + str(i) + " begin ---")
      # Setup the loss fxn
      crossEntrophyWeigths2 = crossEntrophyWeigths.to(device)
      criterion = nn.CrossEntropyLoss(crossEntrophyWeigths2)

      logFile = os.path.join(experimentOutputFolder, experimentExecutionName + '_train.log')

      # Train and evaluate
      model_ft, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist, val_selection_criteria_hist, train_selection_criteria_hist = train_model(model, device, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), logFile = logFile, selection_criteria = selectCriteria, save_all_epochs = calculateAUCROC, experimentExecutionName = experimentExecutionName, auxEnabled = auxEnabled, num_classes = num_classes, experimentOutputFolder = experimentOutputFolder)

      torch.save(model_ft.state_dict(), os.path.join(experimentOutputFolder, experimentExecutionName + '.pth'))

      # validation accuracy
      fig = plt.figure()
      lst = [ x.cpu().item() for x in val_acc_hist ]
      plt.plot(lst)
      ax = plt.gca()
      plt.text(0.05, 0.9, 'FE = ' + str(feature_extract), transform=ax.transAxes)
      plt.text(0.05, 0.8, 'LR = ' + str(learningRate), transform=ax.transAxes)
      plt.text(0.05, 0.7, 'batch = ' + str(batch_size), transform=ax.transAxes)
      plt.suptitle(experimentExecutionName + ' (acc set de validacion)')
      plt.ylabel('Accuracy')
      plt.xlabel('Epochs')
      plt.savefig(os.path.join(experimentOutputFolder, experimentExecutionName + '_val_acc.png'))
      plt.clf()

      # validation loss
      fig = plt.figure()
      plt.plot(val_loss_hist)
      ax = plt.gca()
      plt.text(0.05, 0.3, 'FE = ' + str(feature_extract), transform=ax.transAxes)
      plt.text(0.05, 0.2, 'LR = ' + str(learningRate), transform=ax.transAxes)
      plt.text(0.05, 0.1, 'batch = ' + str(batch_size), transform=ax.transAxes)
      plt.suptitle(experimentExecutionName + ' (loss set de validacion)')
      plt.ylabel('Loss')
      plt.xlabel('Epochs')
      plt.savefig(os.path.join(experimentOutputFolder, experimentExecutionName + '_val_loss.png'))
      plt.clf()

      # train accuracy
      fig = plt.figure()
      lst = [ x.cpu().item() for x in train_acc_hist ]
      ax = plt.gca()
      plt.text(0.05, 0.9, 'FE = ' + str(feature_extract), transform=ax.transAxes)
      plt.text(0.05, 0.8, 'LR = ' + str(learningRate), transform=ax.transAxes)
      plt.text(0.05, 0.7, 'batch = ' + str(batch_size), transform=ax.transAxes)
      plt.plot(lst)
      plt.suptitle(experimentExecutionName + ' (acc set de train)')
      plt.ylabel('Accuracy')
      plt.xlabel('Epochs')
      plt.savefig(os.path.join(experimentOutputFolder, experimentExecutionName + '_train_acc.png'))
      plt.clf()

      # train loss
      fig = plt.figure()
      ax = plt.gca()
      plt.text(0.05, 0.3, 'FE = ' + str(feature_extract), transform=ax.transAxes)
      plt.text(0.05, 0.2, 'LR = ' + str(learningRate), transform=ax.transAxes)
      plt.text(0.05, 0.1, 'batch = ' + str(batch_size), transform=ax.transAxes)
      plt.plot(train_loss_hist)
      plt.suptitle(experimentExecutionName + ' (Loss set de train)')
      plt.ylabel('Loss')
      plt.xlabel('Epochs')
      plt.savefig(os.path.join(experimentOutputFolder, experimentExecutionName + '_train_loss.png'))
      plt.clf()

      # validation selection criteria
      # abbrv = selectCriteriaAbbrv
      # fig = plt.figure()
      # lst = [ x for x in val_selection_criteria_hist ]
      # plt.plot(lst)
      # ax = plt.gca()
      # plt.text(0.05, 0.9, 'FE = ' + str(feature_extract), transform=ax.transAxes)
      # plt.text(0.05, 0.8, 'LR = ' + str(learningRate), transform=ax.transAxes)
      # plt.text(0.05, 0.7, 'batch = ' + str(batch_size), transform=ax.transAxes)
      # plt.suptitle(experimentExecutionName + ' ('+abbrv+' set de validacion)')
      # plt.ylabel(selectCriteria)
      # plt.xlabel('Epochs')
      # plt.savefig(os.path.join(experimentOutputFolder, experimentExecutionName + '_val_'+abbrv+'.png'))
      # plt.clf()

      # # train selection criteria
      # fig = plt.figure()
      # lst = [ x for x in train_selection_criteria_hist ]
      # ax = plt.gca()
      # plt.text(0.05, 0.9, 'FE = ' + str(feature_extract), transform=ax.transAxes)
      # plt.text(0.05, 0.8, 'LR = ' + str(learningRate), transform=ax.transAxes)
      # plt.text(0.05, 0.7, 'batch = ' + str(batch_size), transform=ax.transAxes)
      # plt.plot(lst)
      # plt.suptitle(experimentExecutionName + ' ('+abbrv+' set de train)')
      # plt.ylabel(selectCriteria)
      # plt.xlabel('Epochs')
      # plt.savefig(os.path.join(experimentOutputFolder, experimentExecutionName + '_train_'+abbrv+'.png'))
      # plt.clf()

      stats, accuracy = test_model(model_ft, dataloaders_dict, device, 'val', num_classes)

      print("accuracy: " + str(accuracy))
      accuracyValues.append(accuracy)

      f = open(os.path.join(experimentOutputFolder, experimentExecutionName + "_stats.txt"), "w")
      title = "CN" if num_classes == 3 else "no AD"
      printFile(title + " stats: ", f)
      recall, specificity, precision, f1 = printClassStats(stats[0], f)
      cnStatValues.append({
          "recall": recall,
          "specificity": specificity,
          "precision": precision,
          "f1": f1
      })
      # AD
      printFile("\nAD stats: ", f)
      recall, specificity, precision, f1 = printClassStats(stats[1], f)
      adStatValues.append({
          "recall": recall,
          "specificity": specificity,
          "precision": precision,
          "f1": f1
      })
      if num_classes == 3:
        # MCI
        printFile("\nMCI stats: ", f)
        recall, specificity, precision, f1 = printClassStats(stats[2], f)
        mciStatValues.append({
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1
        })
      f.close()

      print("--- Execution End ---")

    accuracyValues = torch.tensor(accuracyValues)
    std, mean = torch.std_mean(accuracyValues)
    f = open(os.path.join(
        os.path.join(experimentOutputFolder, experimentName + '_results.txt')), "w")
    printFile("Final stats: ", f)
    printFile("Executions: " + str(executions), f)
    printFile("Accuracy mean: " + str(mean.item()), f)
    printFile("Accuracy std: " + str(std.item()), f)
    printFile("Best accuracy: " + str(accuracyValues.max().item()), f)
    printFile("Worst accuracy: " + str(accuracyValues.min().item()), f)
    f.close()
