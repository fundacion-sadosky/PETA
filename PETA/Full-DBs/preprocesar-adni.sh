#!/bin/bash
# Necesario porque el script de Python tiene memory leaks
start_time=$(date +%s)
START=0
STEP=10
MAX=$((3762 + $STEP))
echo $(seq "$START" 3770 "$STEP")
for i in $(seq $START $STEP  $MAX)
do
    echo Procesando desde $i, $STEP imágenes
    python3 preprocesar-adni.py $i $STEP
done

end_time=$(date +%s)

elapsed=$(( end_time - start_time ))

echo "Elapsed time: $elapsed seconds"
