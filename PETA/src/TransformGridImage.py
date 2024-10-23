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
