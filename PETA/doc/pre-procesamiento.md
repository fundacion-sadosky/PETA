# Pre-procesamiento

Este documento describe las etapas de pre procesamiento de las imágenes en la versión mas actualizada del código (puede no corresponder a notebooks y experimentos viejos).

También incluye lo que hace el modelo al cargarlas.

## Preprocesamiento ADNI

- Se bajan todas las imágenes de la página de ADNI.
- Se seleccionan las de resolución uniforme (excluyendo las de 6mm)
- Se ejecuta Full-DBs/preprocesar-adni.py

### preprocesar-adni.py

Este script toma las imågenes y: 
- Las resamplea de 160x160x96 a 128x128x77
- Setea en 0 los valores inferiores a 0 por considerarlos anómalos
- Hace un análisis de la superfície de cada slice resultante de la imågen resampleada. 
- Guarda la imágen nifti procesada en una carpeta con el ID de la imágen junto a un archivo metadata.json que contiene la información de los slices que no deben ser tenidos en cuenta (con la clave "deleteSlices").

## Preprocesamiento Fleni

- Se bajan todas las imágenes de Fleni
- Se consideran solo las que tengan "AC' y "fdg" en el nombre de las carpetas.
- Se convierten las imágenes dcm a formato Nifti.
- Se crea una nueva carpeta con el pet_id como nombre de carpeta, y adentro se pone la imágen nifti. Esto para que en el pre procesamiento se mas fácil identificarlas.
- Se ejecuta Full-DBs/preprocesar-fleni.py

### preprocesar-fleni.py

Simiilar a pre-procesar adni, con la diferencia de que no hace ningún resampling porque las slices de las imágenes ya son de 128x128.

## Procesamiento de imágenes

Esta sección describe el procesamiento de las imágenes al ser cargadas por el modelo para el entrenamiento.

- Para cada imágen, carga el archivo nifti y su correspondiente archivo metadata.json.
- En el caso de ser la etapa de train, realiza el data augmentation.
- Elimina de la imágen los slices que están marcados como "deleteSlices" en el archivo metadata.json.
- Sobre esta nueva imágen elige, con espacio proporcional, 16 slices (redondeando al índice z de la imágen nifti mas cercano).
- A partir de estos slices crea una "grilla" de 16 x 16 slices, de derecha a izquierda y de arriba a abajo, cuya resolución es de 512x512.
- Normaliza esta grilla con los valores de media y std de intensidad del set de entrenamiento de ADNI.
