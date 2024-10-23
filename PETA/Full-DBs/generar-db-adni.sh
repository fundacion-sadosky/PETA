#!/bin/bash
# Divide el set de ADNI en validación y train
# Uso: generar-db-adni.sh <CN> <MCI> <AD> <CARPETA>
# Ejemplo: generar-db-adni.sh 300 1000 200 ADNI-UNIFORM-RESOLUTION
# Genera en la carpeta ADNI-UNIFORM-RESOLUTION los archivos necesarios para entrenar
# Creando un set de train con 300 CN, 1000 MCI/EMCI/LMCI y 200 AD
# Y un set de validacion con el resto de los CN, MCI/EMCI/LMCI y AD que queden
#N_AD=315
#N_CN=539
#N_MCI=1297
N_CN=949 # 854
N_MCI=2147 # 1932
N_AD=555 # 499
# Total: 3651
TRAIN_CN=$1
TRAIN_MCI=$2
TRAIN_AD=$3
VAL_CN=$((N_CN - TRAIN_CN))
VAL_MCI=$((N_MCI - TRAIN_MCI))
VAL_AD=$((N_AD - TRAIN_AD))
TRAIN=$((TRAIN_CN + TRAIN_MCI + TRAIN_AD))
VAL=$((N_AD + N_MCI + N_AD - TRAIN))
NAME=$4
#SOURCE="ADNI-Full-PostProc/ADNI-FULL-PostProc_11_29_2022_UniformResolution.csv"
SOURCE="ADNI-Full-PostProc/ADNI-FULL-PostProc_3_01_2023_UniformResolution.csv"
TYPE="Uniform_Resolution"
CSVT=$NAME/MUESTRA3700_train.csv
CSVV=$NAME/MUESTRA3700_val.csv

if [ -d "$NAME" ]; then
    echo "$NAME existe. Debe ser borrado antes de ejecutar este script"
    exit
fi

echo "Creando carpeta $NAME"
mkdir $NAME

echo "Creando los CSV"
cat $SOURCE | head -n 1 >> $CSVT
cat $SOURCE | head -n 1 >> $CSVV

# CN
regexp="CN"
TRAIN_PER_CAT=$TRAIN_CN
VAL_PER_CAT=$VAL_CN
cat $SOURCE | grep $regexp | head -n $TRAIN_PER_CAT >> $CSVT
cat $SOURCE | grep $regexp | head -n $((TRAIN_PER_CAT + VAL_PER_CAT)) | tail -n $VAL_PER_CAT >> $CSVV
# MCI
TRAIN_PER_CAT=$TRAIN_MCI
regexp="EMCI\|MCI\|LMCI"
VAL_PER_CAT=$((N_MCI - TRAIN_MCI))
cat $SOURCE | grep $regexp | head -n $TRAIN_PER_CAT >> $CSVT
cat $SOURCE | grep $regexp | head -n $((TRAIN_PER_CAT + VAL_PER_CAT)) | tail -n $VAL_PER_CAT >> $CSVV
# AD
regexp="AD"
TRAIN_PER_CAT=$TRAIN_AD
VAL_PER_CAT=$((N_AD - TRAIN_AD))
cat $SOURCE | grep $regexp | head -n $TRAIN_PER_CAT >> $CSVT
cat $SOURCE | grep $regexp | head -n $((TRAIN_PER_CAT + VAL_PER_CAT)) | tail -n $VAL_PER_CAT >> $CSVV



if [[ "$DEBUG" == "1" ]]; then
    echo "Saliendo sin copiar imágenes."
    exit 0
fi

echo "Copiando las imagenes a $NAME"

cat $SOURCE | tail +2 | awk -F ',' '{ print $1  }' | xargs -I{} find ADNI-Full-PostProc -name "*$TYPE*{}*.nii" | xargs -I{} cp -n {} "$NAME"
