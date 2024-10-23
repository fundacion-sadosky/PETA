#!/bin/bash
python3 train.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxlast_2clases_CN,AD.config.js > m8lastCN,AD_train.log 2>&1
python3 train.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxlast_2clases_MCI,AD.config.js > m8lastMCI,AD_train.log 2>&1 
python3 train.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxmost_severe_2clases_CN,AD.config.js > m8sevCN,AD_train.log 2>&1 
python3 train.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxmost_severe_2clases_MCI,AD.config.js > m8sevMCI,AD_train.log 2>&1 
python3 train.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxvisit_2clases_CN,AD.config.js > m8visCN,AD_train.log 2>&1 
python3 train.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxvisit_2clases_MCI,AD.config.js > m8visMCI,AD_train.log 2>&1 

python3 eval.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxlast_2clases_CN,AD.config.js > m8lastCN,AD_eval.log 2>&1 
python3 eval.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxlast_2clases_MCI,AD.config.js > m8lastMCI,AD_eval.log 2>&1 
python3 eval.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxmost_severe_2clases_CN,AD.config.js > m8sevCN,AD_eval.log 2>&1 
python3 eval.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxmost_severe_2clases_MCI,AD.config.js > m8sevMCI,AD_eval.log 2>&1 
python3 eval.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxvisit_2clases_CN,AD.config.js > m8visCN,AD_eval.log 2>&1 
python3 eval.py ../ExperimentosServer/muestraFull3700_8/muestra3700Full_dxvisit_2clases_MCI,AD.config.js > m8visMCI,AD_eval.log 2>&1 
