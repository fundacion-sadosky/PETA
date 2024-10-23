# Configuración
inputFolder = "/Users/hugom/PET-IA/Full-DBs/ebrains/data_nifti_stripped/hc images"
inputFile = "/Users/hugom/PET-IA/Full-DBs/ebrains/sub_info.csv"
outputFolder = "/Users/hugom/Tesis/Imagenes/ebrains-preprocessed"
minSliceSurface= 100.0*100.0
pixelSurface = 2.0*2.0
scale = 1.0 # Esto depende de a qué lo estemos resampleando. Por ejemplo de 160x160x96 a 128x128x77

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
generateRangeOfImages(csv, startImage, min(startImage + step, endImage), inputFolder, outputFolder, minSliceSurface, pixelSurface, scale, sourceExtension = "nii.gz", targetDimX = 128, targetDimY = 128 )
