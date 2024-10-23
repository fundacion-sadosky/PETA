# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys

try:
    from google.colab import drive
    drive.mount('/content/gdrive')
except ModuleNotFoundError:
    print('Not running on Google')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import cv2
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
import json
from torchinfo import summary
from multiprocessing import Pool
import time
import torch.nn.functional as F

# Own
from util import clipped_zoom, printFile, getClassDescription
from transforms import TransformGridImage, ToLabelOutput, ToLabelOutputFleni
from datasets import FleniMyriamDataset, ADNIDataset
from select_criterias import select_criteria_accuracy_aggregate, select_criteria_accuracy_epoch_result, select_criteria_f1AD_aggregate, select_criteria_f1AD_epoch_result
from util import test_model, printClassStats
from config import loadConfig
from train_lib import train_model, set_parameter_requires_grad, initialize_model
from grad_cam import generateGradCAMMask, generateGradCAM

# CAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def testExperiment(config, weights, studyID, calculateHeatmap, expectedClass, occlusionSize, occlusionStride, layerNamesMatrix = [[ "Mixed_7a", "Mixed_7b", "Mixed_7c" ]]):
    imagesFolder = config['imagesFolder']
    fleni60ImagesFolder = config['fleni60ImagesFolder']
    trainDatasetCSV = config['trainDatasetCSV']
    valDatasetCSV = config['valDatasetCSV']
    fleni60ValDatasetCSV = config['fleni60ValDatasetCSV']
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
    trainElements = config['trainElements']
    truthLabel = config['truthLabel']

    if num_classes == 3:
        crossEntrophyWeigths = torch.tensor(trainElements) # Órden: CN, AD, MCI
    else:
      if len(trainElements) == 3:
          crossEntrophyWeigths = torch.tensor([trainElements[0] + trainElements[2], trainElements[1]]) # Órden: CN/MCI, AD
      else: # visit953
          crossEntrophyWeigths = torch.tensor([trainElements[0], trainElements[1]]) # no-AD, AD

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

    selectCriteriaAbbrv = selectCriteriaAbbrv[selectCriteria]

    # End Config
    
    # Init model

    # Detect if we have a GPU available
    device = torch.device(deviceName if torch.cuda.is_available() else "cpu")
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, dropoutRate, auxEnabled, use_pretrained=usePretrained, useGradCAM = True)

    model_state_dict = torch.load(weights, map_location=device)
    model_ft.load_state_dict(model_state_dict)

    model_ft.eval()

    # Print the model we just instantiated
    # print(model_ft)
    #print("Model summary: ")
    #summary(model_ft, input_size=(32, 3, 512, 512))

    # Data augmentation and normalization for training
    # Just normalization for validation

    # Data augmentation and normalization for training
    # Just normalization for validation

    valDatasetCSV = {}

    valGridArgs = {}

    data_transforms = {
        'testSingleElement': transforms.Compose([
            TransformGridImage(),
            transforms.ToTensor(),
            normalizationTransform
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    valDataset = {
        'studyID': [studyID],
        '..': ['..'],    # We won't use this
        'label': ['...'] # We won't use this
    }

    valDataset[truthLabel] = 0 # hacer configurable

    valDataframe = pd.DataFrame(data = valDataset)

    print(valDataframe)

    # Create training and validation datasets

    image_datasets = {
        'testSingleElement': ADNIDataset('valDL', valDataframe, imagesFolder, transform = data_transforms['testSingleElement'], target_transform =ToLabelOutput(num_classes), cacheSize = 0, indexOffset = 0, truthLabel = truthLabel ),
    }

    # Create training and validation dataloaders
    # , 'valFleni'
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0) for x in ['testSingleElement']}
    
    # Eval

    probabilities = None
    predictedClass = None
    
    with torch.no_grad():
        for inputs, labels in dataloaders_dict['testSingleElement']:
            inputs = inputs.to(device)

    inputTensor = None
    target_labels = None
    
    with torch.no_grad():
        for inputs, labels in dataloaders_dict['testSingleElement']:
            inputs = inputs.to(device)

            inputTensor = inputs.squeeze()
            target_labels = labels
            
            outputs = model_ft(inputs)
            
            _, preds = torch.max(outputs, 1)

            predictedClass = int(preds[0].item())

            probabilities = torch.nn.functional.softmax(outputs)

            print(f"Probabilities: {str(probabilities)}")

    # CAM

    # Requerimos todos los gradientes
    # for param in model_ft.parameters():
    #     param.requires_grad = True

    # for layerNames in layerNamesMatrix:
    #     # zero the parameter gradients
    #     #optimizer.zero_grad()

    #     generateGradCAM(model_ft, studyID, inputTensor, layerNames)

    occlusion_size = occlusionSize
    stride = occlusionStride
    
    print(inputTensor.shape)

    if not calculateHeatmap:
        sys.exit(0)
        
    print(f"Occluding with image minimum which is: {inputTensor.min()}")
    imageMin = inputTensor.min()  # Occlude the region with minimum
    
    # Occlusion loop for 3D image
    occlusion_args = []
    for i in range(0, inputTensor.shape[1] - occlusion_size + 1, stride):
        for j in range(0, inputTensor.shape[2] - occlusion_size + 1, stride):
            occlusion_args.append((i, j, outputs, target_labels, model_ft, inputTensor, occlusion_size, device, expectedClass, imageMin))

    print(f"Total occlusions to be calculated: {len(occlusion_args)}")

    calculated_occlusions = 0

    start = time.time()

    sensitivity_map = torch.zeros_like(inputTensor)

    num_workers = 8 # hardcoded
    if torch.cuda.is_available() or num_workers == 1:
        print("Not doing multiple workers b/c it's CUDA or num_workers == 1")
        for occlusion_arg in occlusion_args:
            i, j, impact, _ = calculate_occlusion_wrapper(occlusion_arg)
            sensitivity_map[0:3, i:i+occlusion_size, j:j+occlusion_size] += impact
            # print(f"Processed result for location ({i}, {j}, {k})")
            calculated_occlusions += 1
            # remaining_time = round(time_to_calculate * (len(occlusion_args) - calculated_occlusions) / args.num_workers)
            # Calculamos el remaining time de acuerdo a las oclusiones calculadas
            occlusions_to_calculate = len(occlusion_args) - calculated_occlusions
            elapsed_time = time.time() - start
            remaining_time = round( occlusions_to_calculate * elapsed_time / calculated_occlusions / 60 )
            print(f"Occlusions calculated: {calculated_occlusions}/{len(occlusion_args)}. Remaining time: {remaining_time} minutes")
    else:
        print("Multiprocess processing because it's CPU")
        # Create a pool of worker processes
        with Pool(num_workers) as pool:
            imap_results = pool.imap(calculate_occlusion_wrapper, occlusion_args)

            # Process results in real-time
            for result in imap_results:
                i, j, impact, _ = result
                sensitivity_map[0:3, i:i+occlusion_size, j:j+occlusion_size] += impact
                # print(f"Processed result for location ({i}, {j}, {k})")
                calculated_occlusions += 1
                # remaining_time = round(time_to_calculate * (len(occlusion_args) - calculated_occlusions) / args.num_workers)
                # Calculamos el remaining time de acuerdo a las oclusiones calculadas
                occlusions_to_calculate = len(occlusion_args) - calculated_occlusions
                elapsed_time = time.time() - start
                remaining_time = round( occlusions_to_calculate * elapsed_time / calculated_occlusions / 60 )
                print(f"Occlusions calculated: {calculated_occlusions}/{len(occlusion_args)}. Remaining time: {remaining_time} minutes")

    np.save(f"./{studyID}_original.npy", inputTensor.squeeze().numpy())
    np.save(f"./{studyID}_heatmap.npy", sensitivity_map.numpy())

    return predictedClass, probabilities[0]


def calculate_occlusion(i, j, original_prediction, target_labels, model, img_np_tensor, occlusion_size, device, expectedClass, imageMinimum):
    start = time.time()
    print(f"Occlusion {i}/{j}")
    # Create occluded image
    occluded_img = img_np_tensor.clone()
    occluded_img[0:3, i:i+occlusion_size, j:j+occlusion_size] = imageMinimum 
    occluded_tensor = occluded_img.unsqueeze(0).float().to(device)

    # Make prediction on occluded image
    with torch.no_grad():
        occluded_prediction = model(occluded_tensor).to(device)

    # Compute cross-entropy loss on both predictions
    original_loss = F.cross_entropy(original_prediction, target_labels)
    occluded_loss = F.cross_entropy(occluded_prediction, target_labels)
    # print("original vs occluded loss:")
    # print(original_loss)
    # print(occluded_loss)

    original_prediction = torch.nn.functional.softmax(original_prediction)
    occluded_prediction = torch.nn.functional.softmax(occluded_prediction)

    # print("Original prediction")
    # print(original_prediction)
    # print("Occluded prediction:")
    # print(occluded_prediction)

    # Measure impact (change in cross-entropy loss)
    # Queremos subrayar las regiones donde cambia la loss como positivas
    # Y donde cambia menos como negativas
    # impact = original_loss.item() - occluded_loss.item() # Usar math.abs? Poner en 0? Me importa si al tapar algo ME DISMINUYE la loss?

    # Si son 3 clases esto cambiaría, pero teniendo solo 2 clases la predicción de AD es suficiente
    if expectedClass == 1:
        impact = (original_prediction[0][1] - occluded_prediction[0][1]).item()
    else:
        impact = (original_prediction[0][0] - occluded_prediction[0][0]).item()
    
    print("impact")
    print(impact)
    
    end = time.time()
    # print(f"Time to calculate 1 occlusion: {end - start}")

    time_to_calculate = end - start
    
    return i, j, impact, time_to_calculate

def calculate_occlusion_wrapper(args):
    i, j, impact, time_to_calculate = calculate_occlusion(*args)
    return i, j, impact, time_to_calculate


if __name__ == "__main__":
    if len(sys.argv) != 7 and len(sys.argv) != 4:
        print("Use: ")
        print("test.py configFile weights studyID [expectedClass occlusionSize occlusionStride]")
        sys.exit()
    # Este archivo realiza un eval de UN SOLO ejemplo
    configFile = sys.argv[1]
    weights = sys.argv[2]
    studyID = sys.argv[3]

    calculateHeatmap = False

    if len(sys.argv) == 7:
        expectedClass = int(sys.argv[4])
        occlusionSize = int(sys.argv[5])
        occlusionStride = int(sys.argv[6])
        calculateHeatmap = True
    else:
        expectedClass = None
        occlusionSize = None
        occlusionStride = None

    config = loadConfig(configFile)
    
    predictedClass, probabilities = testExperiment(config, weights, studyID, calculateHeatmap, expectedClass, occlusionSize, occlusionStride)

    print("Predicted class: ", predictedClass)
    print("Confidence: ", probabilities[predictedClass].numpy())
    print("Probs: ", probabilities.numpy())
