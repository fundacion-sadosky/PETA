#!/bin/bash
start_time=$(date +%s)
python3 preprocesar-fleni.py
end_time=$(date +%s)

elapsed=$(( end_time - start_time ))

echo "Elapsed time: $elapsed seconds"
