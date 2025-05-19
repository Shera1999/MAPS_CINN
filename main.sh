
#!/usr/bin/env bash
set -e

source /u/jshera/CL/new_venv/bin/activate
export PYTHONWARNINGS="ignore"

echo "1) Preprocess raw features → processed_data/"
python -m data.cluster_data_filter_simple \
    --unobs_csv data/processed_features.csv \
    --out_dir processed_data

echo "2) Joint SSL + cINN training"
python -m models.train 2>&1 | tee joint_training.log

echo "3) Extract embeddings from JointModel"
python -m postprocessing.generate_embeddings 2>&1 | tee generate_embeddings.log

echo "✅ All done."
