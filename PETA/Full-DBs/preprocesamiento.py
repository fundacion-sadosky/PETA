import dicom2nifti
import nibabel as nib
import nilearn as nil
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import SimpleITK as sitk # para calcular rangos
import numpy as np
import io
from PIL import Image
import random
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
import time
import gc
import sys
import json
from nilearn.image import resample_img

COMPRESSED = True # Creates .nii.gz

# https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image
def fig2img(fig, dpi = 64):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, facecolor='black', edgecolor='none', dpi = dpi, transparent=False) # dpi Requerido para que la imagen sea 512x512
    buf.seek(0)
    img = Image.open(buf)
    return img

# Devuelve un array de índices que deben ser borrados de la imágen Nifti por no cumplir una superfície mínima
def getIndicesToBeDeleted(sample, minSliceSurface = None, pixelSurface = 1.5 * 1.5, minSliceSurfaceThreshold = 0.0):
    # Chequeo de eliminar slices por superfície
    if minSliceSurface == None:
        print("no min slice surface")
        return []
    
    brain_vol_data = sample.get_fdata()
    
    withMoreThan100 = 0
    deleteIndices = []
    if minSliceSurface <= 0.0:
        raise Exception("minSliceSurface should be > 0.0")
    
    for i in range(0, brain_vol_data.shape[2]):
        surface = 0.0
        sliceMatrix = brain_vol_data[:, :, i]
        sliceArray = sliceMatrix.reshape(-1)
        surface =  ( ( sliceArray > minSliceSurfaceThreshold ).sum() ) * pixelSurface
        
        if surface >= minSliceSurface:
            withMoreThan100 += 1
        else:
            deleteIndices.append(i)
            
    if withMoreThan100 < 16:
        raise Exception('No enough brain surface (' + str(withMoreThan100) + ' slices)')
            
    return deleteIndices

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

def transformGridImage(sample, outputImage = None, outputTensor = None, angle = None, zoom = None, shiftX = None, shiftY = None, outputImageDPI = 64,
                        angleTransformChance = 0.1, zoomTransformChance = 0.1, shiftTransformChance = 0.1, 
                        slicesToCut = 0, deleteIndices = [], minSliceSurfaceThreshold = 0.0):
    brain_vol_data = sample.get_fdata()
    fig_rows = 4
    fig_cols = 4
    n_subplots = fig_rows * fig_cols
    
    brain_vol_data = np.delete(brain_vol_data, deleteIndices, axis=2)
            
    # We also replace all values < minSliceSurfaceThreshold with 0
    # In this way we "normalize" the images
    
    n_slice = brain_vol_data.shape[2]

    slices_to_eliminate = slicesToCut

    n_slice_padding = slices_to_eliminate // 2 # quitamos los primeros y ultimos n slices
    n_slice = n_slice - slices_to_eliminate

    step_size = n_slice / n_subplots

    slice_indices = np.arange(n_slice_padding, n_slice_padding + n_slice, step = step_size)

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10], facecolor='black')
    
    if angle == None or angleTransformChance < random.uniform(0.0, 1.0):
        angle = 0.0 # Disable random angle
        
    if zoom != None and random.uniform(0.0, 1.0) > zoomTransformChance:
        zoom = None
        
    if shiftX != None and random.uniform(0.0, 1.0) > shiftTransformChance:
        shiftX = None
        
    if shiftY != None and random.uniform(0.0, 1.0) > shiftTransformChance:
        shiftY = None
        
    idx = 0
    for img in slice_indices:
        processedImage = ndi.rotate(brain_vol_data[:, :, round(img)], 90.0 + angle, mode='constant', reshape = False)
        
        if zoom != None:
            processedImage = clipped_zoom(processedImage, zoom)
        if shiftX != None:
            processedImage = ndi.shift(processedImage, [0.0, shiftX])
        if shiftY != None:
            processedImage = ndi.shift(processedImage, [shiftY, 0.0])
        axs.flat[idx].imshow(np.squeeze(processedImage), cmap='gray', vmin = minSliceSurfaceThreshold) # TODO: this can be wrong for other datasets
        axs.flat[idx].axis('off')
        idx += 1
        
    plt.tight_layout(pad=0.0)

    if outputImage:
            fig.savefig(outputImage, facecolor='black', dpi = 64, transparent=False) # dpi Requerido para que la imagen sea 512x512
            plt.close(fig) # Para que no muestre la imágen
            pyplot.clf()
            plt.cla()
            plt.clf()
            plt.close('all')
            return      
    else:
            image = fig2img(fig, dpi = outputImageDPI)
            plt.close(fig) # Para que no muestre la imágen
            pyplot.clf()
            plt.cla()
            plt.clf()
            plt.close('all')
            return image

# Esta función genera, para una muestra, una serie de archivos incluyendo archivos png, metadata, tensores, etc.
# Esta es la data pre-procesada que será levantada por el script que haga el entrenamiento
def generateImages(sample, sampleId, inputFolder, outputFolder, transformGridParams = { "minSliceSurfaceThreshold": 0.0 }):
    angle15Left = -15.0
    angle15Right = 15.0
    angle08Left = -8.0
    angle08Right = 8.0
    zoomIn = 1.1
    zoomOut = 0.9
    shiftXRight = 10.0
    shiftXLeft = -10.0
    shiftYTop = -10.0
    shiftYBottom = 10.0
    
    # Chequeo de eliminar slices por superfície
    deleteIndices = getIndicesToBeDeleted(sample, **transformGridParams)
    
    sampleFolder = os.path.join(outputFolder, sampleId)
    
    isExist = os.path.exists(sampleFolder)

    minSliceSurfaceThreshold = transformGridParams['minSliceSurfaceThreshold']
    
    if not isExist:
        os.makedirs(sampleFolder)

    if COMPRESSED:
        extension = ".nii.gz"
    else:
        extension = ".nii"
        
    resampledImage = os.path.join(sampleFolder, "resampled-normalized" + extension)
    if not Path(resampledImage).is_file():
        nib.save(sample, resampledImage)
        
    # normal
    normalFile = os.path.join(sampleFolder, "normal.png")
    if not Path(normalFile).is_file():
        transformGridImage(sample, normalFile, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    # # angle15Left
    # angle15LeftFile = os.path.join(sampleFolder, "angle15Left.png")
    # if not Path(angle15LeftFile).is_file():
    #     transformGridImage(sample, angle15LeftFile, angle = angle15Left, angleTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    # # angle15Right
    # angle15RightFile = os.path.join(sampleFolder, "angle15Right.png")
    # if not Path(angle15RightFile).is_file():
    #     transformGridImage(sample, angle15RightFile, angle = angle15Right, angleTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    # # angle8Left
    # angle8LeftFile = os.path.join(sampleFolder, "angle8Left.png")
    # if not Path(angle8LeftFile).is_file():
    #     transformGridImage(sample, angle8LeftFile, angle = angle08Left, angleTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    # # angle8Right
    # angle8RightFile = os.path.join(sampleFolder, "angle8Right.png")
    # if not Path(angle8RightFile).is_file():
    #     transformGridImage(sample, angle8RightFile, angle = angle08Right, angleTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)
    
    # # zoomIn
    # zoomInFile = os.path.join(sampleFolder, "zoomIn.png")
    # if not Path(zoomInFile).is_file():
    #     transformGridImage(sample, zoomInFile, zoom = zoomIn, zoomTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    # # zoomOut
    # zoomOutFile = os.path.join(sampleFolder, "zoomOut.png")
    # if not Path(zoomOutFile).is_file():
    #     transformGridImage(sample, zoomOutFile, zoom = zoomOut, zoomTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    # # shiftXLeft
    # shiftXLeftFile = os.path.join(sampleFolder, "shiftXLeft.png")
    # if not Path(shiftXLeftFile).is_file():
    #     transformGridImage(sample, shiftXLeftFile, shiftX = shiftXLeft, shiftTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)
    
    # # shiftXRight
    # shiftXRightFile = os.path.join(sampleFolder, "shiftXRight.png")
    # if not Path(shiftXRightFile).is_file():
    #     transformGridImage(sample, shiftXRightFile, shiftX = shiftXRight, shiftTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)
    
    # # shiftYTop
    # shiftYTopFile = os.path.join(sampleFolder, "shiftYTop.png")
    # if not Path(shiftYTopFile).is_file():
    #     transformGridImage(sample, shiftYTopFile, shiftY = shiftYTop, shiftTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)
    
    # # shiftYBottom
    # shiftYBottomFile = os.path.join(sampleFolder, "shiftYBottom.png")
    # if not Path(shiftYBottomFile).is_file():
    #     transformGridImage(sample, shiftYBottomFile, shiftY = shiftYBottom, shiftTransformChance = 1.0, deleteIndices = deleteIndices, minSliceSurfaceThreshold = minSliceSurfaceThreshold)

    metadataFile = os.path.join(sampleFolder, "metadata.json")
    if not Path(metadataFile).is_file():
        dict = {
            "sampleId": sampleId,
            "deleteIndices": deleteIndices,
            "minSliceSurfaceThreshold": minSliceSurfaceThreshold,
        }
        with open(metadataFile, "w") as outfile:
            json.dump(dict, outfile)

def resampleNifti(sample, scale = 1/0.8, target_shape=(128, 128, 77), interpolation='nearest'):
    scaleAffine = np.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
   
        [0, 0, 0, 1],
    ])
    newAffine =  sample.affine @ scaleAffine
    # Usamos nearest porque es mas fiel, no inventa valores, aunque se vea peor
    downsampled = resample_img(sample, copy = True, target_affine=newAffine, target_shape=target_shape, interpolation=interpolation)
    return downsampled

# Ídem resample nifti pero tiene un target shape específico para cada dimensión
def resampleNiftiNonIsomorphic(sample, targetDimX, targetDimY):
    currX = sample.header['dim'][1]
    currY = sample.header['dim'][2]
    currZ = sample.header['dim'][3]

    scaleX = 1
    scaleY = 1
    scaleZ = 1

    if targetDimX:
        scaleX = currX / targetDimX

    if targetDimY:
        scaleY = currY / targetDimY

    if scaleY > scaleX:
        scale = scaleY
    else:
        scale = scaleX
    
    scaleAffine = np.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, 1, 0],
   
        [0, 0, 0, 1],
    ])
    newAffine =  sample.affine @ scaleAffine
    # Usamos nearest porque es mas fiel, no inventa valores, aunque se vea peor
    resampled = resample_img(sample, copy = True, target_affine=newAffine, interpolation='nearest')


    # rellenar espacios vacíos si una de las dimensiones no es targetDim

    # Create a new empty array of the desired shape
    new_shape = (targetDimX, targetDimY, currZ)
    new_data = np.zeros(new_shape)

    # Compute the starting and ending indices for the X dimension
    start_x = (new_shape[0] - currX) // 2
    end_x = start_x + currX

    start_y = (new_shape[0] - currY) // 2
    end_y = start_y + currY

    new_data[start_x:end_x, start_y:end_y, :] = sample.get_fdata()

    # Create a new NIfTI image with the resized data
    resized_image = nib.Nifti1Image(new_data, affine=sample.affine)
    
    return resized_image, scale

# Llena con ceros una imágen (hace zoom) hasta que tenga la dimensión correcta
# def resizeImage(sample, targetDimX, targetDimY, targetDimZ):
    

def normalizeRange(sample, vmin = 0.0):
    intensities = sample.get_fdata()
    intensities = np.maximum(vmin, intensities)
    mn, mx = intensities.min(), intensities.max()
    rescaled = (intensities - mn) / (mx - mn)

    new_img = nib.Nifti1Image(rescaled, sample.affine, sample.header)
    return new_img

def removeSmallValues(sample, vmin = 0.0):
    intensities = sample.get_fdata()
    intensities = np.maximum(vmin, intensities)
    
    new_img = nib.Nifti1Image(intensities, sample.affine, sample.header)
    return new_img

# Esto es para generar las imágenes en batch ya que por algún motivo Python empieza a consumir toda la memoria
# Posiblemente nibabel o numpy tengan un memory leak
def generateRangeOfImages(csv, start, end, inputFolder, outputFolder, minSliceSurface, pixelSurface, scale, sourceExtension = "nii", targetDimX = None, targetDimY = None, targetDimZ = None, onLoadImageHook = None, target_shape = (128, 128, 77), downsampleInterpolation = 'nearest'):
    imagesWithErrors = []
    print("Generating images from " + str(start) + " to " + str(end))
    resampledPixelSurface = pixelSurface / (scale*scale) # NOTA: acá se podría pensar que debería ser al cubo, pero no, porque es superfície
    if scale != 1.0:
        print("Resampled pixel surface: ", resampledPixelSurface)
    for i in range(start, end):
        studyID = csv.iloc[i, 0]
        print("Progress: " + str(i) + "/" + str(len(csv)) + "stID: " + studyID)
        rglob = '*'+str(studyID)+'*.' + sourceExtension

        for path in Path(inputFolder).rglob(rglob):
            filename = str(path)
            brain_vol = nib.load(filename)

            if onLoadImageHook:
                # Hook to process images
                brain_vol = onLoadImageHook(brain_vol)

            # Se resamplea y se normalizan los valores
            if scale != 1.0: # downsampling
                resampledNifti = resampleNifti(brain_vol, 1.0/scale, target_shape, interpolation = downsampleInterpolation)
            else:
                resampledNifti = brain_vol

            pixelFinalSurface = resampledPixelSurface
                
            if targetDimX != None or targetDimY != None or targetDimZ != None: # upsampling
                # TODO: hacer ejemplo y revisar si está bien
                pixDimX = brain_vol.header['pixdim'][1]
                pixDimY = brain_vol.header['pixdim'][2]
                resampledNifti, resampledScale = resampleNiftiNonIsomorphic(resampledNifti, targetDimX, targetDimY)
                # Pixel surface can change if we change dimensions, so it needs to be recalculated
                print(f"Resampled scale: {resampledScale}")

                print(resampledNifti.header['pixdim'][1])
                print(resampledNifti.header['pixdim'][2])
                pixelFinalSurface = resampledNifti.header['pixdim'][1] * resampledNifti.header['pixdim'][2]

            print(f"Final pixel surface: {pixelFinalSurface}")            
                
            #newRangedNifti = normalizeRange(resampledNifti)

            resampledNifti = removeSmallValues(resampledNifti)
            
            # Ejecutar todo de vuelta pero sin generar las imágenes? Chequear que todas tengan slices suficientes!!
            try:
                generateImages(resampledNifti, studyID, inputFolder, outputFolder, transformGridParams = { "pixelSurface": pixelFinalSurface, "minSliceSurface": minSliceSurface, "minSliceSurfaceThreshold": 0.0  })
            except Exception as e:
                print(f"Error generating images for {studyID}: {str(e)}")
                imagesWithErrors.append(studyID)
            brain_vol.uncache()
            del brain_vol
            if i != 0 and i % 20 == 0:
                gc.collect()
                
    return imagesWithErrors

# Las imágenes de Fleni ya están en un formato studyId.nii en una carpeta
# Además no necesitan ser resampleadas
def generateRangeOfImagesFleni(inputFolder, outputFolder, minSliceSurface, pixelSurface, scale, sourceExtension = "nii"):
    rglob = f"*.{sourceExtension}"
    processed = 0
    imagesWithErrors = []
    for path in Path(inputFolder).rglob(rglob):
        filename = str(path)
        filenamWOExtension = path.name[:-(len(sourceExtension) + 1)]
        studyID = filenamWOExtension
        print(f"Processing {filenamWOExtension} ({processed + 1})")
        brain_vol = nib.load(filename)

        newRangedNifti = removeSmallValues(brain_vol)

        try:
            generateImages(newRangedNifti, filenamWOExtension, inputFolder, outputFolder, transformGridParams = { "pixelSurface": pixelSurface, "minSliceSurface": minSliceSurface, "minSliceSurfaceThreshold": 0.0  })
        except Exception as e:
            print(f"Error generating images for {studyID}: {str(e)}")
            imagesWithErrors.append(studyID)

        processed += 1

    return processed, imagesWithErrors

def convertToNiiGz(inputFolder):
    rglob = '*.nii'
    for path in Path(inputFolder).rglob(rglob):
        filename = str(path)
        filenamWOExtension = path.stem
        print("Processing " + filenamWOExtension)
        newFileName = filename[:-3]
        brain_vol = nib.load(filename)

        nib.save(brain_vol, os.path.join(inputFoldder, newFileName))
