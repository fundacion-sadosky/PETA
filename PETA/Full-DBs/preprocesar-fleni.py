# Configuración
inputFolder = "/Users/hugom/Downloads/pet_fleni_imgs_a_stripear"
outputFolder = "/Users/hugom/Tesis/Imagenes/fleni-stripped-preprocessed4"
minSliceSurface= 100.0*100.0
pixelSurface = 2.0*2.0
scale = 1.0 # Para Fleni NO resampleamos, es ignorado.

# Ejecución
import sys
from preprocesamiento import generateRangeOfImagesFleni

processed, failedImages = generateRangeOfImagesFleni(inputFolder, outputFolder, minSliceSurface, pixelSurface, scale, sourceExtension = "nii.gz" )

print(f"Failed images ({len(failedImages)  / processed}): ")
print(failedImages)
