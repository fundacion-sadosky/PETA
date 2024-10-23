#!/bin/bash

python3 eval.py ../ExperimentosServer/muestraFull3700_9/muestra3700Full_dxlast_3clases.config.js > m9last3clases.evalnew 2>&1; 

python3 eval.py ../ExperimentosServer/muestraFull3700_9/muestra3700Full_dxmost_severe_3clases.config.js  > m9sev3clases.evalnew  2>&1; 

python3 eval.py ../ExperimentosServer/muestraFull3700_9/muestra3700Full_dxvisit_3clases.config.js  > m9visit3clases.evalnew  2>&1; 

python3 eval.py ../ExperimentosServer/muestraFull3700_9/muestra3700Full_dxlast_2clases.config.js >  m9last2clases.evalnew 2>&1 ; 

python3 eval.py ../ExperimentosServer/muestraFull3700_9/muestra3700Full_dxmost_severe_2clases.config.js  > m9sev2clases.evalnew  2>&1; 

python3 eval.py ../ExperimentosServer/muestraFull3700_9/muestra3700Full_dxvisit_2clases.config.js > m9visit2clases.evalnew  2>&1;
