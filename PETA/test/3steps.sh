#!/bin/bash
cd src
# Training model
echo "Training..."
python3 train.py config.train-test.json
# Evaluationg model
echo "Evaluating..."
python3 eval.py config.train-test.json
# Testing model


echo "3 steps test finished successfully"
