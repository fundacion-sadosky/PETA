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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if not torch.cuda.is_available():
    raise Exception("CUDA is not available")

# Config

imagesFolder = '/content/gdrive/MyDrive/Tesis/Imagenes/ADNI-MUESTRA-FULL-stripped-preprocessed3'
fleniImagesFolder = '/content/gdrive/MyDrive/Tesis/Imagenes/fleni-stripped-preprocessed3'
trainDatasetCSV = '/content/gdrive/MyDrive/Tesis/Imagenes/Muestra3700_80_10_10_train.csv'
valDatasetCSV =   '/content/gdrive/MyDrive/Tesis/Imagenes/Muestra3700_80_10_10_val.csv'
fleniValDatasetCSV =   '/content/gdrive/MyDrive/Tesis/Imagenes/fleni-stripped-preprocessed/match-curated.csv'
experimentName = 'MuestraFull3700_4_server_acc'
experimentOutputFolder = '/content/gdrive/MyDrive/Tesis/Experimentos/muestraFull3700_4'
experimentDescription = 'Guardar todas las epochs (optativo). Mostrar AUC/ROC'
executions = 1

# Config 2

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32

dl_num_workers = 4

# Number of epochs to train for
num_epochs = 25

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

usePretrained = True

# Habilita la salida auxiliar
auxEnabled = True

learningRate = 0.0001
dropoutRate = 0.6
if num_classes == 3:
  crossEntrophyWeigths = torch.tensor([759.0,444.0,1717.0]) # Órden: CN, AD, MCI
else:
  crossEntrophyWeigths = torch.tensor([759.0 + 1717.0,444.0]) # Órden: CN/MCI, AD

trainMean = 0.1716601789041244 #preproc3, < 0s eliminados
trainStd = 0.3936839672084841 #preproc3
#trainMean = 0.1534203209139499  #preproc4, sin eliminar < 0s
#trainStd =  0.4048895150096513   #preproc4
normalization = {
  #"trainMeans": [0.485, 0.456, 0.406], # ImageNet
  #"trainStds": [0.229, 0.224, 0.225].  # ImageNet
  "trainMeans": [trainMean, trainMean, trainMean],
  "trainStds": [trainStd, trainStd, trainStd]
}

deviceName = 'cuda:0'

# Data augmentation
dataAugmentation = {
    "angle": 8,
    "shiftX": 10,
    "shiftY": 10,
    "zoom": 0.1,
    "shear": np.pi / 16,
}
# dataAugmentation = {}

selectCriteria = "f1AD"

validationCacheSize = 0
trainCacheSize = 0

calculateAUCROC = True # For this is necessary to save all epochs?

# Debug
debug = False
doTrain = True # True unless we want to skip training

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
f.write("cross entrophy weights: " + str(crossEntrophyWeigths) + "\n")
f.write("dataAugmentation: " + str(json.dumps(dataAugmentation)) + "\n")
f.write("selectCriteria: " + str(selectCriteria) + "\n")
f.write("executions: " + str(executions) + "\n")
f.write("normalization: " + str(json.dumps(normalization)) + "\n")
f.write("deviceName: " + str(deviceName) + "\n")
f.write("validationCacheSize: " + str(validationCacheSize) + "\n")
f.write("trainCacheSize: " + str(trainCacheSize) + "\n")
f.write("calculateAURROC: " + str(calculateAUCROC) + "\n")
f.close()


f = open(os.path.join(experimentOutputFolder, experimentName + "_descripcion.txt"), "w")
f.write(experimentDescription)
f.close()


# Utils

selectCriteriaAbbrv = {
    "accuracy": "acc",
    "f1AD": "f1AD",
}[selectCriteria]

def logDebug(str):
    if debug:
        print(str)


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
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

class LoadImage(object):
    """Loads an image
    """

    def __init__(self, imagesFolder):
      self.imagesFolder = imagesFolder

    def __call__(self, studyID):
      imageFile = os.path.join(self.imagesFolder, studyID, "resampled-normalized.nii.gz")
      metadataFile = os.path.join(self.imagesFolder, studyID, "metadata.json")
      f = open(metadataFile, "r")
      metadata = json.load(f)
      f.close()

      return nib.load(imageFile), metadata

class TransformGridImage():
    def __init__(self, angle = None, zoom = None, shiftX = None, shiftY = None, shear = None):
      self.angle = angle
      self.zoom = zoom
      self.shiftX = shiftX
      self.shiftY = shiftY
      self.shear = shear
      self.sliceWidth = 128
      self.sliceHeight = 128
      self.filters = []
      if (self.zoom):
        self.filters.append("zoom")
      if (self.angle):
        self.filters.append("angle")
      if self.shiftX:
        self.filters.append("shiftX")
      if self.shiftY:
        self.filters.append("shiftY")
      if self.shear:
        self.filters.append("shear")

    def __call__(self, studyData):
      sample = studyData[0]
      metadata = studyData[1]

      brain_vol_data = sample.get_fdata()
      fig_rows = 4
      fig_cols = 4
      n_subplots = fig_rows * fig_cols

      deleteIndices = metadata["deleteIndices"]
    
      brain_vol_data = np.delete(brain_vol_data, deleteIndices, axis=2)
    
      n_slice = brain_vol_data.shape[2]

      step_size = n_slice / n_subplots

      slice_indices = np.arange(0, n_slice, step = step_size)

      channels = 3
      grid = np.empty( shape = (fig_rows * 128, fig_cols * 128, channels), dtype=np.float32)

      angle = 0.0
      zoom = None
      shiftX = None
      shiftY = None
      shear = None

      filter = None
      if len(self.filters) > 0:
        filter = random.choice(self.filters)
    
      if filter == "angle":
        angle = random.uniform(-self.angle, self.angle)
      elif filter == "zoom":
        zoom = 1.0 + random.uniform(-self.zoom, self.zoom)
      elif filter == "shiftX":
        shiftX = round(128 * random.uniform(-self.shiftX, self.shiftX) / 100.0)
      elif self.shiftY == "shiftY":
        shiftY = round(128 * random.uniform(-self.shiftY, self.shiftY) / 100.0)
      elif filter == "shear":
        shear = random.uniform(-self.shear, self.shear)
        
      slice_index = 0
      for i in range(0, fig_rows):
        for j in range(0, fig_cols):
            slice_index  = slice_indices[i * fig_rows + j]
            processedImage = ndi.rotate(brain_vol_data[:, :, round(slice_index)], 90.0 + angle, mode='nearest', reshape = False)

            if zoom != None:
                processedImage = clipped_zoom(processedImage, zoom, mode = 'nearest')

            if shiftX != None:
                processedImage = ndi.shift(processedImage, [0.0, shiftX], mode = 'nearest')

            if shiftY != None:
                processedImage = ndi.shift(processedImage, [shiftY, 0.0], mode = 'nearest')

            if shear != None:
                # shear debe estar en radianes
                # https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/affine_transformations.py#L348
                transform = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
                processedImage = ndi.affine_transform(processedImage,
                    transform,
                    #offset=(0, -self.sliceHeight//2, 0),
                    output_shape=(self.sliceWidth, self.sliceHeight))

            rowStart = i * 128
            rowEnd = (i + 1) * 128
            colStart = j * 128
            colEnd = (j+1)*128
            
            # 3 channels
            for c in range(0, channels):
              grid[rowStart:rowEnd, colStart:colEnd, c] = processedImage.copy()

            slice_index += 1

      return grid


class ToLabelOutput(object):
    def __init__(self, numClasses = 3):
        self.numClasses = numClasses

    def __call__(self, label):
        if self.numClasses == 3:
          if label == "CN":
            return 0
          elif label == "AD":
            return 1
          else:
            return 2 # MCI, LMCI, EMC
        else:
          if label == "AD":
            return 1
          else:
            return 0 # CN/MCI collapsed class


class ToLabelOutputFleni(object):
    def __init__(self, numClasses = 3):
        self.numClasses = numClasses

    def __call__(self, label):
      if int(label) == 1:
        return 1 # AD
      elif int(label) == 0:
        if self.numClasses == 3: 
          return 2 # MCI
        else:
          return 0 # CN/MCI collapsed class
      else:
        raise Exception("Wrong Fleni label: " + str(label))


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
                 cacheSize = 200, indexOffset = 1):
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
        self.indexOffset = indexOffset

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
          label = self.csv.iloc[idx, 2 + self.indexOffset]

          image, metadata = self.loadImage(studyID)
          self.storeInCache(idx, image, label, metadata)
        

        if self.transform:
            image = self.transform([image, metadata])
            
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def printFile(text, file):
  print(text)
  if file != None:
      file.write(text + "\n")

# Modelo

def select_criteria_accuracy_aggregate(running_corrects, outputs, labels):
  _, preds = torch.max(outputs, 1)
  return running_corrects + torch.sum(preds == labels.data)

def select_criteria_accuracy_epoch_result(running_corrects, dataset):
  return running_corrects.double() / len(dataset)


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

def select_criteria_f1AD_epoch_result(aggregate, dataset):
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


def train_model(model, dataloaders, criterion, optimizer, experimentExecutionName, num_epochs=25, is_inception=True, logFile = None, selection_criteria = "accuracy", save_all_epochs = False):
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
            epoch_selection_criteria = select_criteria_epoch_result(selection_criteria_aggregate, dataloaders[phase].dataset)

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


# Initialize and reshape inception

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
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

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=usePretrained)

# Print the model we just instantiated
print(model_ft)


