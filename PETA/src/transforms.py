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
from nilearn.image import resample_img

# Own
from util import clipped_zoom, printFile

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

# Used by the new model
class TransformReduced3DImage():
    def __init__(self, angle = None, zoom = None, shiftX = None, shiftY = None, shear = None, shiftZ = None, collapseYDim = None):
      # shiftZ is ignored. It's only there to have compatibility with Transform3DImage
      # same for collapseYDim
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
      grid = np.empty( shape = (128, 128, fig_rows * fig_cols), dtype=np.float32)

      angle = 0.0
      zoom = None
      shiftX = None
      shiftY = None
      shear = None

      filter = None
      if len(self.filters) > 0:
        filter = random.choice(self.filters)
    
      if filter == "angle":
        # print("angle")
        angle = random.uniform(-self.angle, self.angle)
      elif filter == "zoom":
        # print("zoom")
        zoom = 1.0 + random.uniform(-self.zoom, self.zoom)
      elif filter == "shiftX":
        # print("shiftX")
        shiftX = round(128 * random.uniform(-self.shiftX, self.shiftX) / 100.0)
      elif self.shiftY == "shiftY":
        # print("shiftY")
        shiftY = round(128 * random.uniform(-self.shiftY, self.shiftY) / 100.0)
      elif filter == "shear":
        # print("shear")
        shear = random.uniform(-self.shear, self.shear)
        
      slice_index = 0
      for i in range(0, fig_rows):
        for j in range(0, fig_cols):
            slice_index  = slice_indices[i * fig_rows + j]
            processedImage = ndi.rotate(brain_vol_data[:, :, round(slice_index)], 90.0 + angle, mode='nearest', reshape = False)
            # print("rotated image")
            # print(processedImage.shape)
            # print(processedImage)
            if zoom != None:
                processedImage = clipped_zoom(processedImage, zoom, mode = 'nearest')
                # print(f"zoomed image. zoom = {zoom}")
                # print(processedImage.shape)
                # print(processedImage)

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
            
            # 1 channel
            
            grid[:, :, i*fig_rows + j] = processedImage.copy()

            slice_index += 1

      # Hago esto porque así parece requerirlo el modelo
      return grid.transpose((0, 2, 1))

# Ídem resample nifti pero tiene un target shape específico para cada dimensión
def resampleNiftiNonIsomorphic(sample, scaleX = 1, scaleY = 1, scaleZ = 1, target_shape = None):
    currX = sample.header['dim'][1]
    currY = sample.header['dim'][2]
    currZ = sample.header['dim'][3]
    
    scaleAffine = np.array([
        [scaleX, 0, 0, 0],
        [0, scaleY, 0, 0],
        [0, 0, scaleZ, 0],
   
        [0, 0, 0, 1],
    ])
    newAffine =  sample.affine @ scaleAffine
    # Usamos nearest porque es mas fiel, no inventa valores, aunque se vea peor
    resampled = resample_img(sample, copy = True, target_affine=newAffine, interpolation='nearest', target_shape = target_shape)
    return resampled

# Used by the new model
class Transform3DImage():
    def __init__(self, angle = None, zoom = None, shiftX = None, shiftY = None, shiftZ = None, shear = None, yDim = 16, augmentation = 'one', resampleZ = False, collapseYDim = None, collapseYDimChance = 1.0, verbose = False):
      self.angle = angle
      self.zoom = zoom
      self.shiftX = shiftX
      self.shiftY = shiftY
      self.shiftZ = shiftZ
      self.shear = shear
      self.collapseYDim = collapseYDim
      self.collapseYDimChance = collapseYDimChance
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
      if self.shiftZ:
        self.filters.append("shiftZ")
      if self.collapseYDim:
        self.filters.append("collapseYDim")

      self.yDim = yDim # altura
      self.augmentation = augmentation # augmentation strategy

      self.resampleZ = resampleZ

      if resampleZ and resampleZ != 1.0 and collapseYDim:
          raise Exception("Transform3DImage] resampleZ y collapseYDim no deberían usarse juntos")

      self.verbose = verbose

    def __call__(self, studyData):
      sample = studyData[0]
      metadata = studyData[1]

      brain_vol_data = sample.get_fdata()
      fig_rows = 4
      fig_cols = 4
      n_subplots = fig_rows * fig_cols

      # deleteIndices = metadata["deleteIndices"]
      # print("Delete indices")
      # print(deleteIndices)
    
      # # brain_vol_data = np.delete(brain_vol_data, deleteIndices, axis=2)

      # print("Brain vol data shape: ")
      # print(brain_vol_data.shape)

      if self.resampleZ and self.resampleZ != 1.0:
          if self.verbose:
              print(f"Transform3DImage] Resample Z to {self.resampleZ}")
          # print(brain_vol_data.shape)
          sample = resampleNiftiNonIsomorphic(sample, scaleZ = self.resampleZ)
          brain_vol_data = sample.get_fdata()

          # print("Resampling to")
          # print(brain_vol_data.shape)

      angle = 0.0
      zoom = None
      shiftX = None
      shiftY = None
      shiftZ = None
      shear = None

      if self.augmentation == 'one':
          filter = None
          if len(self.filters) > 0:
              filter = random.choice(self.filters)

          if self.verbose:
              print(f"Transform3DImage] Filter chosen: {filter}")

          if filter == "angle":
              # print("angle")
            angle = random.uniform(-self.angle, self.angle)
          elif filter == "zoom":
              # print("zoom")
            zoom = 1.0 + random.uniform(-self.zoom, self.zoom)
          elif filter == "shiftX":
              # print("shiftX")
            shiftX = round(128 * random.uniform(-self.shiftX, self.shiftX) / 100.0)
          elif self.shiftY == "shiftY":
              # print("shiftY")
            shiftY = round(128 * random.uniform(-self.shiftY, self.shiftY) / 100.0)
          elif self.shiftZ == "shiftZ":
              # print("shiftZ")
            shiftZ = round(self.yDim * random.uniform(-self.shiftZ, self.shiftZ) / 100.0)
          elif filter == "shear":
              # print("shear")
            shear = random.uniform(-self.shear, self.shear)
          elif filter == "collapseYDim":
              y_dim = brain_vol_data.shape[2]
              # Le bajamos la resolución y se la subimos de vuelta
              if self.verbose:
                  print(f"Transform3DImage] collapseYDim to {self.collapseYDim}")
              sample = resampleNiftiNonIsomorphic(sample, scaleZ = y_dim / self.collapseYDim, target_shape = (128, 128, self.collapseYDim))
              if self.verbose:
                  print(f"Transform3DImage] Resampling again to {y_dim}")
              sample = resampleNiftiNonIsomorphic(sample, scaleZ = self.collapseYDim / y_dim, target_shape = (128, 128, y_dim))
              brain_vol_data = sample.get_fdata()
              
                
      elif self.augmentation == 'all':
          angle = random.uniform(-self.angle, self.angle)
          zoom = 1.0 + random.uniform(-self.zoom, self.zoom)
          shiftX = round(128 * random.uniform(-self.shiftX, self.shiftX) / 100.0)
          shiftY = round(128 * random.uniform(-self.shiftY, self.shiftY) / 100.0)
          if self.shiftZ:
              shiftZ = round(self.yDim * random.uniform(-self.shiftZ, self.shiftZ) / 100.0)
          shear = random.uniform(-self.shear, self.shear)
          if self.collapseYDim and random.random() < self.collapseYDimChance:
              y_dim = brain_vol_data.shape[2]
              # Le bajamos la resolución y se la subimos de vuelta
              sample = resampleNiftiNonIsomorphic(sample, scaleZ = y_dim / self.collapseYDim, target_shape = (128, 128, self.collapseYDim))
              sample = resampleNiftiNonIsomorphic(sample, scaleZ = self.collapseYDim / y_dim, target_shape = (128, 128, y_dim))
              brain_vol_data = sample.get_fdata()


      # Se hace después porque puede ser que collapseYDim se ejecute y nos quede una imágen mas chica
      n_slice = brain_vol_data.shape[2]

      # print(brain_vol_data.shape)

      # print("n_slice")
      # print(n_slice)
      # print("yDim")
      # print(self.yDim)
      if n_slice > self.yDim:
          raise Exception(f"Transform3DImage] Invalid shape: {str(brain_vol_data.shape)}. shape[2] should be less than {self.yDim}")

      min_value = brain_vol_data.min()



      # slice_indices = np.linspace(0, n_slice - 1, num = self.yDim)
      # print(slice_indices)
      # print(brain_vol_data.shape)

      grid = np.empty( shape = (128, 128, self.yDim), dtype=np.float32)
      grid.fill(min_value)

      # print("Grid shape:")
      # print(grid.shape)

      # Ojo porque lo de abajo está deformando los cerebros
      # Habría que rehacerlo para que los centre Y PROBAR QU´E PASA

      first_pos = (grid.shape[2] - n_slice) // 2
      # print(grid.shape)
      # print(n_slice)
      # print(first_pos)

      if shiftZ:
          first_pos = first_pos + shiftZ

      for i in range(0, n_slice):
        processedImage = ndi.rotate(brain_vol_data[:, :, i], 90.0 + angle, mode='nearest', reshape = False)
        
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

        # 1 channel

        grid[:, :, i + first_pos] = processedImage.copy()
        
      # slice_index = 0
      # for i in range(0, self.yDim):
      #   slice_index  = slice_indices[i]
      #   processedImage = ndi.rotate(brain_vol_data[:, :, round(slice_index)], 90.0 + angle, mode='nearest', reshape = False)
        
      #   if zoom != None:
      #       processedImage = clipped_zoom(processedImage, zoom, mode = 'nearest')

      #   if shiftX != None:
      #       processedImage = ndi.shift(processedImage, [0.0, shiftX], mode = 'nearest')

      #   if shiftY != None:
      #       processedImage = ndi.shift(processedImage, [shiftY, 0.0], mode = 'nearest')

      #   if shear != None:
      #       # shear debe estar en radianes
      #       # https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/affine_transformations.py#L348
      #       transform = np.array([[1, -np.sin(shear), 0],
      #                        [0, np.cos(shear), 0],
      #                        [0, 0, 1]])
      #       processedImage = ndi.affine_transform(processedImage,
      #           transform,
      #           #offset=(0, -self.sliceHeight//2, 0),
      #           output_shape=(self.sliceWidth, self.sliceHeight))

      #   # 1 channel

      #   grid[:, :, i] = processedImage.copy()

      #   slice_index += 1

      # Hago esto porque así parece requerirlo el modelo
      return grid.transpose((0, 2, 1))

# Reduce una imágen de 128x128x77 a una imágen de 128x128xn, usando nearest
class ReduceImage(object):
    def __init__(self, new_dimension_y):
        self.new_dimension_y = new_dimension_y

    def __call__(self, img):
        data = img.get_fdata()
        # Target dimensions (128x128x47)
        target_slices = 47

        # Calculate the step size for selecting slices
        step_size = data.shape[2] // target_slices

        # Initialize an empty array for the downsampled data
        downsampled_data = np.zeros((data.shape[0], data.shape[1], target_slices))

        # Fill the downsampled data by repeating nearest slices
        for i in range(target_slices):
            nearest_slice_index = min((i + 1) * step_size, data.shape[2]) - 1
            # print("nearest slice index", nearest_slice_index)
            downsampled_data[:, :, i] = data[:, :, nearest_slice_index]

        # Create a new NIfTI image using the downsampled data
        downsampled_img = nib.Nifti1Image(downsampled_data, img.affine)

        return downsampled_img

# Deja los tensores con valores entre [min, max]
class MinMaxNormalization(object):
    def __init__(self, min, max):
        self.new_min = min
        self.new_max = max

    def __call__(self, v):
        v_min, v_max = v.min(), v.max()
        return (v - v_min)/(v_max - v_min)*(self.new_max - self.new_min) + self.new_min


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

class ToLabelOutputEBRAINS(object):
    def __call__(self, label):
        return 0 # siempre es CN

class ToLabelOutputConfigurable(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary[label]
