# keras-training

## Installation
Install `miniconda2` by sourcing `install_miniconda.sh` in your home directory. Log out and log back in after this.
```bash
cp install_miniconda.sh ~/
cd ~
source install_miniconda.sh
```

Install the rest of the dependencies:
```bash
cd ~/keras-training
source install.sh
```

Each time you log in set things up:
```bash
source setup.sh
```

## Conversion of data
All of the data ntuple files are available here: https://cernbox.cern.ch/index.php/s/AgzB93y3ac0yuId

To add the truth values and flatten the trees (you can skip this step)
```bash
cd ~/keras-training/convert
python addTruth.py -t t_allpar \
../data/processed-pythia82-lhc13-*-pt1-50k-r1_h022_e0175_t220_nonu.root
```

To `hadd` these files and convert from `TTree` to `numpy array` with
random shuffling (you can skip this step)
```bash
hadd -f \
../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.root \
../data/processed-pythia82-lhc13-*-pt1-50k-r1_h022_e0175_t220_nonu_truth.root
python convert.py -t t_allpar_new \
../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.root
```

You can also copy this file directly from: https://cernbox.cern.ch/index.php/s/aGjXWDrDpugHeMf

## Training and evaluation
To run a simple training:
```bash
cd ~/keras-training/train
python train.py -t t_allpar_new \
-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-o train_simple/
```

and evaluate the training:
```bash
python eval.py -t t_allpar_new \
-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-m train_simple/KERAS_check_best_model.h5 \
-o eval_simple /
```

## Pruning and retraining
To prune the trained model by removing weights below a certain
threshold (relative weight < 0.2):
```bash
mkdir prune_simple_relwmax2e-1
python prune.py train_simple/KERAS_check_best_model.h5 \
--relative-weight-max 2e-1 \
-o prune_simple_relwmax2e-1/pruned_model.h5
```

and evaluate the pruned model:
```bash
python eval.py -t t_allpar_new \
-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-m prune_simple_relwmax2e-1/pruned_model.h5 \
-o eval_simple_relwmax2e-1/
```

To retrain the pruned model (keeping the pruned weights fixed to 0):
```bash
python retrain.py -t t_allpar_new \
-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-o retrain_simple_relwmax2e-1 \
-m prune_simple_relwmax2e-1/pruned_model.h5  \
-d prune_simple_relwmax2e-1/pruned_model_drop_weights.h5
```

and evaluate the pruned, retrained model:
```bash
python eval.py -t t_allpar_new \
-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-m retrain_simple_relwmax2e-1/KERAS_check_best_model.h5 \
-o eval_retrain_simple_relwmax2e-1/
```


