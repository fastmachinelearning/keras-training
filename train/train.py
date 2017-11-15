import sys
import os
import keras
import numpy as np
from optparse import OptionParser
import h5py
from keras.optimizers import Adam, Nadam
from callbacks import all_callbacks
import pandas as pd
from keras.layers import Input
# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    (options,args) = parser.parse_args()

    if os.path.isdir(options.outputDir):
        raise Exception('output directory must not exists yet')
    else:
        os.mkdir(options.outputDir)
    

    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    print treeArray.dtype.names
    
    # List of features to use
    features = ['j_zlogz', 'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt',
                'j_d2_b1_mmdt', 'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt',
                'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_mass_mmdt', 'j_multiplicity']
    
    # List of labels to use
    labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']

    # Convert to dataframe
    features_df = pd.DataFrame(treeArray,columns=features)
    labels_df = pd.DataFrame(treeArray,columns=labels)
    
    # Convert to numpy array with correct shape
    features_val = features_df.values
    labels_val = labels_df.values
    print features_val.shape
    print labels_val.shape

    from models import three_layer_model

    keras_model = three_layer_model(Input(shape=(features_val.shape[1],)), labels_val.shape[1])

    startlearningrate=0.0001
    adam = Adam(lr=startlearningrate)
    keras_model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

        
    callbacks=all_callbacks(stop_patience=1000, 
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001, 
                            lr_cooldown=2, 
                            lr_minimum=0.0000001,
                            outputDir=options.outputDir)

    keras_model.fit(features_val, labels_val, batch_size = 1024, epochs = 100,
                    validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)
