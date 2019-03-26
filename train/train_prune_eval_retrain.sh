#!/bin/bash

python train.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -c train_config_threelayer.yml \
       -o train_simple_l10p0001/

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m train_simple_l10p0001/KERAS_check_best_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001/

mkdir -p prune_simple_l10p0001_33perc
python prune.py -m train_simple_l10p0001/KERAS_check_best_model.h5 \
       --relative-weight-percentile 32.7 \
       -o prune_simple_l10p0001_33perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_33perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_33perc/

python retrain.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_33perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -d prune_simple_l10p0001_33perc/pruned_model_drop_weights.h5 \
       -o retrain_simple_l10p0001_33perc/

mkdir -p prune_simple_l10p0001_48perc
python prune.py -m retrain_simple_l10p0001_33perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 47.5 \
       -o prune_simple_l10p0001_48perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_48perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_48perc/

python retrain.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_48perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -d prune_simple_l10p0001_48perc/pruned_model_drop_weights.h5 \
       -o retrain_simple_l10p0001_48perc/

mkdir -p prune_simple_l10p0001_57perc
python prune.py -m retrain_simple_l10p0001_48perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 56.5 \
       -o prune_simple_l10p0001_57perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_57perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_57perc/

python retrain.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_57perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -d prune_simple_l10p0001_57perc/pruned_model_drop_weights.h5 \
       -o retrain_simple_l10p0001_57perc/

mkdir -p prune_simple_l10p0001_63perc
python prune.py -m retrain_simple_l10p0001_57perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 63.0 \
       -o prune_simple_l10p0001_63perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_63perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_63perc/

python retrain.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_63perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -d prune_simple_l10p0001_63perc/pruned_model_drop_weights.h5 \
       -o retrain_simple_l10p0001_63perc/

mkdir -p prune_simple_l10p0001_67perc
python prune.py -m retrain_simple_l10p0001_63perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 66.7 \
       -o prune_simple_l10p0001_67perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_67perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_67perc/

python retrain.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_67perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -d prune_simple_l10p0001_67perc/pruned_model_drop_weights.h5 \
       -o retrain_simple_l10p0001_67perc/

mkdir -p prune_simple_l10p0001_70perc
python prune.py -m retrain_simple_l10p0001_67perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 70.0 \
       -o prune_simple_l10p0001_70perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_70perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_70perc/

python retrain.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_70perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -d prune_simple_l10p0001_70perc/pruned_model_drop_weights.h5 \
       -o retrain_simple_l10p0001_70perc/

mkdir -p prune_simple_l10p0001_72perc
python prune.py -m retrain_simple_l10p0001_70perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 71.7 \
       -o prune_simple_l10p0001_72perc/pruned_model.h5

python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_72perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_72perc/
