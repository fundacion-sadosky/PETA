#!/bin/bash
# Copia todas las imágenes nifti de cierto tipo a una carpeta
NAME="$1"
TYPE="Uniform_Resolution"
cat ADNI-Full-PostProc/ADNI-FULL-PostProc_11_29_2022_UniformResolution.csv | tail +2 | awk -F ',' '{ print $1  }' | xargs -I{} find ADNI-Full-PostProc -name "*$TYPE*{}*.nii" | xargs -I{} cp -n {} "$NAME" 
