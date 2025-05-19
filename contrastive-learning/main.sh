#!/usr/bin/env bash
source /u/jshera/CL/new_venv/bin/activate
export PYTHONWARNINGS="ignore"

echo "1) Preprocessing raw features → processed_data/"
python /u/jshera/MAPS_CINN/data/cluster_data_filter_simple.py \
  --unobs_csv /u/jshera/MAPS_CINN/data/processed_features.csv \
  --out_dir processed_data

echo "2) Joint SSL + cINN training"
python /u/jshera/MAPS_CINN/models/train.py 2>&1 | tee joint_training.log

echo "3) Extract embeddings from trained encoder"
python /u/jshera/MAPS_CINN/postprocessing/generate_embeddings.py 2>&1 | tee generate_embeddings.log

echo "=== All done! ==="
