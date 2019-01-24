from __future__ import print_function
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
from sklearn import preprocessing
import yaml
import models

# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'w')
    jsonString = keras_model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

def get_features(options, yamlConfig):
    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    print(treeArray.shape)
    print(treeArray.dtype.names)
    
    # List of features to use
    features = yamlConfig['Inputs']
    
    # List of labels to use
    labels = yamlConfig['Labels']

    # Convert to dataframe
    features_labels_df = pd.DataFrame(treeArray,columns=list(set(features+labels)))
    features_labels_df = features_labels_df.drop_duplicates()

    features_df = features_labels_df[features]
    labels_df = features_labels_df[labels]
    
    if 'Conv' in yamlConfig['InputType']:
        labels_df = labels_df.drop_duplicates()
        
    # Convert to numpy array 
    features_val = features_df.values
    labels_val = labels_df.values     

    if 'j_index' in features:
        features_val = features_val[:,:-1] # drop the j_index feature
    if 'j_index' in labels:
        labels_val = labels_val[:,:-1] # drop the j_index label
        print(labels_val.shape)

    if yamlConfig['InputType']=='Conv1D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['MaxParticles'], len(features)-1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index']==labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values
            #features_val_i = features_val[index_values[0]:index_values[-1]+1,:-1] # drop the last feature j_index
            features_val_i = features_val[np.array(index_values),:]
            nParticles = len(features_val_i)
            #print("before", features_val_i[:,0])
            features_val_i = features_val_i[features_val_i[:,0].argsort()[::-1]] # sort descending by first value (ptrel, usually)
            #print("after", features_val_i[:,0])
            if nParticles>yamlConfig['MaxParticles']:
                features_val_i =  features_val_i[0:yamlConfig['MaxParticles'],:]
            else:        
                features_val_i = np.concatenate([features_val_i, np.zeros((yamlConfig['MaxParticles']-nParticles, len(features)-1))])
            features_2dval[i, :, :] = features_val_i

        features_val = features_2dval

    elif yamlConfig['InputType']=='Conv2D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['BinsX'], yamlConfig['BinsY'], 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index']==labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values
            
            xbins = np.linspace(yamlConfig['MinX'],yamlConfig['MaxX'],yamlConfig['BinsX']+1)
            ybins = np.linspace(yamlConfig['MinY'],yamlConfig['MaxY'],yamlConfig['BinsY']+1)

            x = features_df_i[features[0]]           
            y = features_df_i[features[1]]
            w = features_df_i[features[2]]

            hist, xedges, yedges = np.histogram2d(x, y, weights=w, bins=(xbins,ybins))

            for ix in range(0,yamlConfig['BinsX']):
                for iy in range(0,yamlConfig['BinsY']):
                    features_2dval[i,ix,iy,0] = hist[ix,iy]
        features_val = features_2dval

    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2, random_state=42)
    
    #Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']!='Conv1D' and yamlConfig['InputType']!='Conv2D':
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)

    #Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']!='Conv1D' and yamlConfig['InputType']!='Conv2D' and yamlConfig['KerasLoss']=='squared_hinge':
        scaler = preprocessing.MinMaxScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)
        y_train_val = y_train_val * 2 - 1
        y_test = y_test * 2 - 1
        
    #Normalize conv inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']=='Conv1D':
        reshape_X_train_val = X_train_val.reshape(X_train_val.shape[0]*X_train_val.shape[1],X_train_val.shape[2])
        scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
        for p in range(X_train_val.shape[1]):
            X_train_val[:,p,:] = scaler.transform(X_train_val[:,p,:])
            X_test[:,p,:] = scaler.transform(X_test[:,p,:])    

    if 'j_index' in labels:
        labels = labels[:-1]

    return X_train_val, X_test, y_train_val, y_test, labels

## Config module
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

if __name__ == "__main__":
    parser = OptionParser()
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

    X_train_val, X_test, y_train_val, y_test, labels  = get_features(options, yamlConfig)
    
    #from models import three_layer_model
    model = getattr(models, yamlConfig['KerasModel'])    
    if 'L1RegR' in yamlConfig:
        keras_model = model(Input(shape=X_train_val.shape[1:]), y_train_val.shape[1], l1Reg=yamlConfig['L1Reg'], l1RegR=yamlConfig['L1RegR'] )
    else:
        keras_model = model(Input(shape=X_train_val.shape[1:]), y_train_val.shape[1], l1Reg=yamlConfig['L1Reg'] )

    print_model_to_json(keras_model,options.outputDir + '/' + 'KERAS_model.json')

    startlearningrate=0.0001
    adam = Adam(lr=startlearningrate)
    keras_model.compile(optimizer=adam, loss=[yamlConfig['KerasLoss']], metrics=['accuracy'])

        
    callbacks=all_callbacks(stop_patience=1000, 
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001, 
                            lr_cooldown=2, 
                            lr_minimum=0.0000001,
                            outputDir=options.outputDir)

    keras_model.fit(X_train_val, y_train_val, batch_size = 1024, epochs = 500,
                    validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)
