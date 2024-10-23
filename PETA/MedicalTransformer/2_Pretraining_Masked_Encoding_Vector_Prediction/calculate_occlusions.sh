#!/bin/bash

OCCLUSION_SIZE=$1
STRIDE=$2

echo AD
for STUDY_ID in I128833 I47403 I146553 I1334685 I85844 ; do
    echo $STUDY_ID
    python3 inference.py --axial_slicelen 119 --weights log/visit953adniv2-clf/model/7_Multiview_MEP_CN_ResNet_freeze_16.pt --img ../../../ADNI-MUESTRA-FULL-stripped-preprocessed3/$STUDY_ID/resampled-normalized.nii.gz --occlusion-size $OCCLUSION_SIZE --occlusion-stride $STRIDE --num_workers 4 --calculate-heatmap --expected-class 1 --save-heatmap $STUDY_ID-size$OCCLUSION_SIZE-stride$STRIDE
done

echo no AD
for STUDY_ID in I225971 I321658 I77185 I325148 I223643 ; do
    echo $STUDY_ID
    python3 inference.py --axial_slicelen 119 --weights log/visit953adniv2-clf/model/7_Multiview_MEP_CN_ResNet_freeze_16.pt --img ../../../ADNI-MUESTRA-FULL-stripped-preprocessed3/$STUDY_ID/resampled-normalized.nii.gz --occlusion-size $OCCLUSION_SIZE --occlusion-stride $STRIDE --num_workers 4 --calculate-heatmap --expected-class 0 --save-heatmap $STUDY_ID-size$OCCLUSION_SIZE-stride$STRIDE
done
