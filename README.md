# keras-training

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

You can also copy this file directly from: `root://cmseos.fnal.gov//eos/uscms/store/user/woodson/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z`

To run a simple training:
```bash
cd ~/keras-training/train
python train.py -t t_allpar_new -i \
../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-o train_simple/
```

and evaluate the training:
```bash
python eval.py -t t_allpar_new -i \
../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
-m train_simple/KERAS_check_best_model.h5 \
-o eval_simple /
```
