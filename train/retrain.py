import sys
import os
import keras
import numpy as np
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
from optparse import OptionParser
import h5py
from keras.optimizers import Adam, Nadam
from callbacks import all_callbacks
import pandas as pd
from keras.layers import Input
from sklearn.model_selection import train_test_split
import yaml
from train import parse_config
import models

# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--model'   ,action='store',type='string',dest='inputModel'   ,default='prune_simple/pruned_model.h5', help='input model')
    parser.add_option('-d','--drop-weights'   ,action='store',type='string',dest='dropWeights'   ,default='prune_simple/pruned_model_drop_weights.h5', help='dropped weights h5 file')
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-c','--config'   ,action='store',type='string', dest='config', default='train_config_threelayer.yml', help='configuration file')
    (options,args) = parser.parse_args()
     
    yamlConfig = parse_config(options.config)

    if os.path.isdir(options.outputDir):
        raise Exception('output directory must not exists yet')
    else:
        os.mkdir(options.outputDir)    

    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    print treeArray.dtype.names
    
    # List of features to use
    features = yamlConfig['Inputs']
    
    # List of labels to use
    labels = yamlConfig['Labels']

    # Convert to dataframe
    features_df = pd.DataFrame(treeArray,columns=features)
    labels_df = pd.DataFrame(treeArray,columns=labels)
    
    # Convert to numpy array with correct shape
    features_val = features_df.values
    labels_val = labels_df.values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2, random_state=42)
    print X_train_val.shape
    print y_train_val.shape
    print X_test.shape
    print y_test.shape

    #from models import three_layer_model_constraint
    model_constraint = getattr(models, yamlConfig['KerasModelRetrain'])

    # Instantiate new model with added custom constraints
    keras_model = model_constraint(Input(shape=(X_train_val.shape[1],)), y_train_val.shape[1], l1Reg=yamlConfig['L1Reg'], h5fName = options.dropWeights )

    startlearningrate=0.0001
    adam = Adam(lr=startlearningrate)
    keras_model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

    # Load pre-trained weights!
    keras_model.load_weights(options.inputModel, by_name=True)
        
    callbacks=all_callbacks(stop_patience=1000, 
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001, 
                            lr_cooldown=2, 
                            lr_minimum=0.0000001,
                            outputDir=options.outputDir)

    keras_model.fit(X_train_val, y_train_val, batch_size = 1024, epochs = 100,
                    validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)
