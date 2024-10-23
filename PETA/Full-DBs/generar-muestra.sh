#!/bin/bash
# Genera un directorio de muestra para ser usado por los scripts del modelo.
# Ejemplo: generar-muestra.sh 90 30 ADNI-MUESTRA-120 generara una carpeta ADNI-MUESTRA-120
# Con 90 items de entrenamiento y 30 de validacion.
TRAIN=$1
VAL=$2
NAME=$3
SOURCE="ADNI-Full-PostProc/ADNI-FULL-PostProc_11_29_2022_UniformResolution.csv"
TYPE="Uniform_Resolution"

TRAIN_PER_CAT=$((TRAIN / 3))
VAL_PER_CAT=$((VAL / 3))


CSVT=$NAME/MUESTRA_train.csv
CSVV=$NAME/MUESTRA_val.csv

if [ -d "$NAME" ]; then
    echo "$NAME existe. Debe ser borrado antes de ejecutar este script"
    exit
fi

echo "Creando carpeta $NAME"
mkdir $NAME

echo "Creando los CSV"
cat $SOURCE | head -n 1 >> $CSVT
cat $SOURCE | head -n 1 >> $CSVV

for diagnosis in "CN" "MCI" "AD"; do
  regexp=$diagnosis
  if [ "$diagnosis" == "MCI"  ]; then
      regexp="EMCI\|MCI\|LMCI"
  fi
  cat $SOURCE | grep $regexp | head -n $TRAIN_PER_CAT >> $CSVT
  cat $SOURCE | grep $regexp | head -n $((TRAIN_PER_CAT + VAL_PER_CAT)) | tail -n $VAL_PER_CAT >> $CSVV
done

if [[ "$DEBUG" == "1" ]]; then
    echo "Saliendo sin copiar imágenes."
    exit 0
fi

echo "Copiando las imagenes a $NAME"

echo "Copiando imagenes de entrenamiento"
cat $CSVT | tail -n $TRAIN | awk -F ',' '{ print $1 }' | sed 's/"//g' | xargs -I{} find ADNI-Full-PostProc -name "*$TYPE*{}*.nii" | xargs -I{} cp {} $NAME

echo "Copiando imagenes de test"
cat $CSVV | tail -n $VAL | awk -F ',' '{ print $1 }' | sed 's/"//g' | xargs -I{} find ADNI-Full-PostProc -name "*$TYPE*{}*.nii" | xargs -I{} cp {} $NAME


CANT_IMG=$(find $NAME -name "*.nii" | wc -l)
echo "Imágenes copiadas: $CANT_IMG"
