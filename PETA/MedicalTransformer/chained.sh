#!/bin/bash
set -e
export WANDB_API_KEY="f35fdf6d3e9c68879928fad9696248e617133d3f"

PATTERN="END OF TRAINING"

PREFIX=last2 # visit953, last2
TRUTH_LABEL=last # visit, last
PRETRAIN_SAVE_EVERY=1000

# Ejecuta los 3 casos con pre-entrenamiento

# Nota: en --collapse-ydim 0.5 el 0.5 en realidad es arbitrario. Solo habilita el filtro.

check_stop_file() {
    # Check if the "stop" file exists in the current directory
    if [ -e "stop" ]; then
        echo "Stopping script execution because 'stop' file exists."
        exit 1  # Exit with a non-zero status to indicate an error
    fi
}

# Pre-entrenando solo con adni
# export NAME=${PREFIX}adniv2

# echo Pretrain

# cd 1_Pretraining_Triplet_Loss

# python3 prepretrain.py --batch_size 32 --epoch 300 --axial_slicelen 119 --dataset adni --augmentation one --num_workers 8 --lr 0.0001 --steplr-step-size 1 --steplr-gamma 0.99 --eval-dataset adni --seed 22 --eval-shuffle 1 --nonlinear-prob 0.5 --eval-rounds 5 --save-every $PRETRAIN_SAVE_EVERY --normalization "min-max" --name $NAME-cnn --truth-label $TRUTH_LABEL --subset_size 1024 --with-replacement --wandb --eval-fleni --collapse-ydim 0.5 > $NAME-cnn.log 2>&1 &

# wait

# check_stop_file

# cd ..

# echo TXs

# cd 2_Pretraining_Masked_Encoding_Vector_Prediction

# python3 pretrain_ae.py --batch_size 16 --epoch 300 --axial_slicelen 119 --dataset adni --augmentation one --num_workers 8 --seed 22 --normalization min-max --prepretrain-weights="/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/$NAME-cnn/model/prepretrain_model.pt" --lr 0.01 --wandb --mask_ratio 0.1 --name $NAME-tx --patience 50 --save-every $PRETRAIN_SAVE_EVERY --truth-label $TRUTH_LABEL --with-replacement --subset_size 1024 --steplr-gamma 1  --eval-fleni --collapse-ydim 0.5 > $NAME-tx.log 2>&1 &

# wait

# LOG_FILE="$NAME-tx.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

# echo Clf

# python3 classification.py --axial_slicelen 119 --epoch 150 --batch_size 32 --augmentation one --num_workers 8 --lr 0.0001 --name $NAME-clf --seed 22 --wandb --normalization "min-max" --pretrain-weights "./log/$NAME-tx/model/pretrain_model.pt" --save-all --truth-label $TRUTH_LABEL --eval-fleni --collapse-ydim 0.5 > $NAME-clf.log 2>&1 &

# wait

# LOG_FILE="$NAME-clf.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

# cd ..

# check_stop_file

# # Pre-entrenando con merida y con chinese
# export NAME=${PREFIX}plusv2

# echo Pretrain

# cd 1_Pretraining_Triplet_Loss

# python3 prepretrain.py --batch_size 32 --epoch 300 --axial_slicelen 119 --dataset adni-plus --augmentation one --num_workers 8 --lr 0.0001 --steplr-step-size 1 --steplr-gamma 0.99 --eval-dataset adni --seed 22 --eval-shuffle 1 --nonlinear-prob 0.5 --eval-rounds 5 --save-every $PRETRAIN_SAVE_EVERY --normalization "min-max" --name $NAME-cnn --truth-label $TRUTH_LABEL --subset_size 1024 --with-replacement --wandb  --eval-fleni --collapse-ydim 0.5 > $NAME-cnn.log 2>&1 &

# wait

# LOG_FILE="$NAME-cnn.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

# cd ..

# check_stop_file

# echo TXs

# cd 2_Pretraining_Masked_Encoding_Vector_Prediction

# python3 pretrain_ae.py --batch_size 16 --epoch 300 --axial_slicelen 119 --dataset adni-plus --augmentation one --num_workers 8 --seed 22 --normalization min-max --prepretrain-weights="/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/$NAME-cnn/model/prepretrain_model.pt" --lr 0.01 --wandb --mask_ratio 0.1 --name $NAME-tx --patience 50 --save-every $PRETRAIN_SAVE_EVERY --truth-label $TRUTH_LABEL --with-replacement --subset_size 1024 --steplr-gamma 1 --eval-fleni --collapse-ydim 0.5 --verbose > $NAME-tx.log 2>&1 &

# wait

# LOG_FILE="$NAME-tx.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

# echo Clf

# python3 classification.py --axial_slicelen 119 --epoch 150 --batch_size 32 --augmentation one --num_workers 8 --lr 0.0001 --name $NAME-clf --seed 22 --wandb --normalization "min-max" --pretrain-weights "./log/$NAME-tx/model/pretrain_model.pt" --save-all --truth-label $TRUTH_LABEL --eval-fleni --collapse-ydim 0.5 > $NAME-clf.log 2>&1 &

# wait

# LOG_FILE="$NAME-clf.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

# cd ..


# Pre-entrenando con todos
export NAME=${PREFIX}allv2

# echo Pretrain

# cd 1_Pretraining_Triplet_Loss

# python3 prepretrain.py --batch_size 32 --epoch 300 --axial_slicelen 119 --dataset all --augmentation one --num_workers 8 --lr 0.0001 --steplr-step-size 1 --steplr-gamma 0.99 --eval-dataset adni --seed 22 --eval-shuffle 1 --nonlinear-prob 0.5 --eval-rounds 5 --save-every $PRETRAIN_SAVE_EVERY --normalization "min-max" --name $NAME-cnn --truth-label $TRUTH_LABEL --subset_size 1024 --with-replacement --wandb --eval-fleni --collapse-ydim 0.5 > $NAME-cnn.log 2>&1 &

# wait

# LOG_FILE="$NAME-cnn.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

# cd ..

# echo TXs

cd 2_Pretraining_Masked_Encoding_Vector_Prediction

# python3 pretrain_ae.py --batch_size 16 --epoch 300 --axial_slicelen 119 --dataset all --augmentation one --num_workers 8 --seed 22 --normalization min-max --prepretrain-weights="/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/$NAME-cnn/model/prepretrain_model.pt" --lr 0.01 --wandb --mask_ratio 0.1 --name $NAME-tx --patience 50 --save-every $PRETRAIN_SAVE_EVERY --truth-label $TRUTH_LABEL --with-replacement --subset_size 1024 --steplr-gamma 1 --eval-fleni --collapse-ydim 0.5 --verbose > $NAME-tx.log 2>&1 &

# wait

# LOG_FILE="$NAME-tx.log"
# if grep -q "$PATTERN" "$LOG_FILE";
#  then
#      echo "Task ended correctly"
#  else
#      echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
#      echo "Exiting..."
#      exit 1
# fi

echo Clf

python3 classification.py --axial_slicelen 119 --epoch 150 --batch_size 32 --augmentation one --num_workers 8 --lr 0.0001 --name $NAME-clf --seed 22 --wandb --normalization "min-max" --pretrain-weights "./log/$NAME-tx/model/pretrain_model.pt" --save-all --truth-label $TRUTH_LABEL --eval-fleni  --collapse-ydim 0.5 > $NAME-clf.log 2>&1 &

wait

LOG_FILE="$NAME-clf.log"
if grep -q "$PATTERN" "$LOG_FILE";
 then
     echo "Task ended correctly"
 else
     echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
     echo "Exiting..."
     exit 1
fi

cd ..

check_stop_file

cd 2_Pretraining_Masked_Encoding_Vector_Prediction

# Sin pre-entrenamiento
export NAME=${PREFIX}v2

echo Clf

python3 classification.py --axial_slicelen 119 --epoch 150 --batch_size 32 --augmentation one --num_workers 8 --lr 0.0001 --name $NAME-clf --seed 22 --wandb --normalization "min-max" --save-all --truth-label $TRUTH_LABEL --eval-fleni --collapse-ydim 0.5 > $NAME-clf.log 2>&1 &

wait

LOG_FILE="$NAME-clf.log"
if grep -q "$PATTERN" "$LOG_FILE";
 then
     echo "Task ended correctly"
 else
     echo "Error: The Pattern '$PATTERN' was NOT Found in '$LOG_FILE'"
     echo "Exiting..."
     exit 1
fi

cd ..

check_stop_file

echo The End
