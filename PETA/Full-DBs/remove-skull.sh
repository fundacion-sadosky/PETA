#!/bin/bash
# Toma una carpeta con archivos nifti y recursivamente elimina los skulls de dicha carpeta,
# creando otra carpeta con el mismo nombre y "-stripped" al final

SOURCE=$1
OUTPUT="$SOURCE-stripped"
PROCESSES=4

rm -r "$OUTPUT"

echo "Generando $OUTPUT"

cp -r "$SOURCE" "$OUTPUT"

FILE_NUMBER=$(find $OUTPUT -name "*.nii" | wc -l)

echo "Convirtiendo $FILE_NUMBER imágenes"

find "$OUTPUT" -name "*.nii" | parallel --bar -P $PROCESSES mri_synthstrip -i {} -o {}
