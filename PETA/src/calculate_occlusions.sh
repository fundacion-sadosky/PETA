#!/bin/bash

STRIDE=16
OCCLUSION_SIZE=16

echo AD
for STUDY_ID in I128833 I47403 I146553 I1334685 I85844 ; do
    echo $STUDY_ID
    python3 test.py ../ExperimentosServer/muestra953_1/muestra953.config.test.json  ../server-backup/muestra953_1/muestra953_1_dxvisit953_2classes_0_epoch36.pth $STUDY_ID 1 $OCCLUSION_SIZE $STRIDE
done

echo no AD
for STUDY_ID in I225971 I321658 I77185 I325148 I223643 ; do
    echo $STUDY_ID
    python3 test.py ../ExperimentosServer/muestra953_1/muestra953.config.test.json  ../server-backup/muestra953_1/muestra953_1_dxvisit953_2classes_0_epoch36.pth $STUDY_ID 0 $OCCLUSION_SIZE $STRIDE
done
