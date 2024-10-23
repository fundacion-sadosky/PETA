#!/bin/bash

python3 train.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxlast_3clases.config.js > m10last3clases; 

python3 train.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxmost_severe_3clases.config.js  > m10sev3clases; 

python3 train.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxvisit_3clases.config.js  > m10visit3clases; 

python3 train.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxlast_2clases.config.js >  m10last2clases; 

python3 train.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxmost_severe_2clases.config.js  > m10sev2clases; 

python3 train.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxvisit_2clases.config.js > m10visit2clases;



python3 eval.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxlast_3clases.config.js > m10last3clases.eval; 

python3 eval.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxmost_severe_3clases.config.js  > m10sev3clases.eval; 

python3 eval.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxvisit_3clases.config.js  > m10visit3clases.eval; 

python3 eval.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxlast_2clases.config.js >  m10last2clases.eval; 

python3 eval.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxmost_severe_2clases.config.js  > m10sev2clases.eval; 

python3 eval.py ../ExperimentosServer/muestraFull3700_10/muestra3700Full_dxvisit_2clases.config.js > m10visit2clases.eval;
