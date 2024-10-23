#!/bin/bash
# Calculates the stats of a classifier log file


FILE=$1
echo "Fleni 60 mean AUC:"
cat $FILE | grep "fleni60 Loss" | awk '{ x += $9 }END{ print x/NR }'
echo "Fleni 100 mean AUC:"
cat $FILE | grep "fleni100 Loss" | awk '{ x += $9 }END{ print x/NR }'
echo "Fleni 600 test mean AUC:"
cat $FILE | grep "fleni600_test Loss" | awk '{ x += $9 }END{ print x/NR }'


# STD:
# awk '{sum+=$9; sumsq+=$9*$9}END{print sqrt( (1/(NR-1) * (sumsq - ( (sum * sum)/NR )) ))}'

echo "Fleni 60 STD AUC:"
cat $FILE | grep "fleni60 Loss" | awk '{sum+=$9; sumsq+=$9*$9}END{print sqrt( (1/(NR-1) * (sumsq - ( (sum * sum)/NR )) ))}'
echo "Fleni 100 STD AUC:"
cat $FILE | grep "fleni100 Loss" | awk '{sum+=$9; sumsq+=$9*$9}END{print sqrt( (1/(NR-1) * (sumsq - ( (sum * sum)/NR )) ))}'
echo "Fleni 600 test STD AUC:"
cat $FILE | grep "fleni600_test Loss" | awk '{sum+=$9; sumsq+=$9*$9}END{print sqrt( (1/(NR-1) * (sumsq - ( (sum * sum)/NR )) ))}'
