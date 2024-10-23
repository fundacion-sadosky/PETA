# Experimentos

Esta carpeta contiene los notebooks de varios scripts de experimentos.

## ADNI

El script mas actualizado para ejecutar la muestra completa de ADNI es [](./MuestraFull_3.ipynb).

En la sección Configuración del notebook se pueden setear los distintos parámetros.

### Parámetros generales:

- imagesFolder: carpeta donde están las imágenes pre procesadas.
- trainDatasetCSV: CSV con el set de entrenamiento.
- valDatasetCSV: CSV con el set de validación.
- experimentName: los archivos generados tendrán este prefijo. Sirve para distinguir diferentes ejecuciones de un mismo experimento (por ejemplo ejecutar el mismo código pero con variaciones de los parámetros)
- experimentOutputFolder: El script, al ejecutarse, creará unos cuantos archivos en una carpeta destino. Esto incluye un archivo con los pesos, los gráficos, los logs, un archivo de métricas y archivos que describen el experimento.
- experimentDescription: este string se guardará en el archivo `<experimentName>_description.txt`.
- executions: para algunos experimentos, si queremos un test estadístico, permite ejecutar varias veces el mismo experimento y sacar media y std del accuracy.

### Parámetros del modelo

Estos incluyen el batch\_size, num\_epochs, etc.
