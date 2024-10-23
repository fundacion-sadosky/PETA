# Configuración
# inputFolder = "/Users/hugom/PET-IA/Full-DBs/Chinese/Normalized-images"
inputFolder = "/Users/hugom/PET-IA/Full-DBs/ines-merida-db-stripped/"
inputFile = "/Users/hugom/PET-IA/Full-DBs/ines-merida-db/pool/DM/TEP/CERMEP_MXFDG/BASE/DATABASE_SENT/ALL/participants.tsv" # 
outputFolder = "/Users/hugom/Tesis/Imagenes/merida-preprocessed"
minSliceSurface= 100.0*100.0
pixelSurface = 1.0
scale = 0.5268  # 243 -> 128
target_shape = (128, 128, 119) # 243x243x226 / 0.5268

# Ejecución
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from preprocesamiento import generateRangeOfImages

startImage=int(sys.argv[1])
step=int(sys.argv[2]) # step is the number of images every process will process

csv = pd.read_table(inputFile, index_col=False)

print(len(csv))

endImage = len(csv)

# Agregamos a la primera dimensión 18 espacios vacíos de cada lado
# Conversión de 207 a 243
def onLoadImageHook(brain_vol):
    data = brain_vol.get_fdata()
    new_vol = np.zeros((data.shape[1], data.shape[1], data.shape[2]))
    diff = data.shape[1] - data.shape[0]

    diff_2 = diff // 2
    right_limit = brain_vol.shape[0] + diff_2
    new_vol[diff_2:right_limit, :, :] = data.copy()

    return nib.Nifti1Image(new_vol, brain_vol.affine, header = brain_vol.header)


print("Procesando desde %d hasta %d"%(startImage, min(startImage + step, endImage)))
generateRangeOfImages(csv, startImage, min(startImage + step, endImage), inputFolder, outputFolder, minSliceSurface, pixelSurface, scale, sourceExtension = "nii.gz", onLoadImageHook = onLoadImageHook, target_shape = target_shape)
