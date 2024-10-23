# Configuración
inputFolder = "/Users/hugom/Tesis/Imagenes/ADNI-MUESTRA-FULL-stripped"
inputFile = "/Users/hugom/PET-IA/Full-DBs/ADNI-Full-PostProc/ADNI-FULL-PostProc_3_01_2023_UniformResolution.csv"
outputFolder = "/Users/hugom/Tesis/Imagenes/ADNI-MUESTRA-FULL-stripped-preprocessed-linear"
minSliceSurface= 100.0*100.0
pixelSurface = 1.5*1.5
scale = 0.8 # Esto depende de a qué lo estemos resampleando. Por ejemplo de 160x160x96 a 128x128x77

# Ejecución
import sys
import pandas as pd
from preprocesamiento import generateRangeOfImages

startImage=int(sys.argv[1])
step=int(sys.argv[2]) # step is the number of images every process will process

csv = pd.read_csv(inputFile)

print(len(csv))

endImage = len(csv)

print("Procesando desde %d hasta %d"%(startImage, min(startImage + step, endImage)))
generateRangeOfImages(csv, startImage, min(startImage + step, endImage), inputFolder, outputFolder, minSliceSurface, pixelSurface, scale, downsampleInterpolation = 'linear' )
