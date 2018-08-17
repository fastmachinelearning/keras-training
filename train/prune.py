from __future__ import print_function
import sys
import os
import math
from optparse import OptionParser
from keras.models import load_model, Model
from argparse import ArgumentParser
from keras import backend as K
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
import pandas as pd
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
from constraints import ZeroSomeWeights
from train import print_model_to_json
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})


# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
def getWeightArray(model):
    allWeights = []
    allWeightsNonRel = []
    allWeightsByLayer = {}
    allWeightsByLayerNonRel = {}
    for layer in model.layers:         
        if layer.__class__.__name__ in ['Dense', 'Conv1D', 'LSTM']:
            original_w = layer.get_weights()
            weightsByLayer = []
            weightsByLayerNonRel = []
            for my_weights in original_w:                
                if len(my_weights.shape) < 2: # bias term, ignore for now
                    continue
                #l1norm = tf.norm(my_weights,ord=1)
                elif len(my_weights.shape) == 2: # Dense or LSTM
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                elif len(my_weights.shape) == 3: # Conv1D
                    # (filter_width, n_inputs, n_filters)
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_0 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_reduce_max_0,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                with tf.Session():
                    #l1norm_val = float(l1norm.eval())
                    tensor_max = float(tensor_reduce_max_2.eval())
                it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readwrite'])   
                while not it.finished:
                    w = it[0]
                    allWeights.append(abs(w)/tensor_max)
                    allWeightsNonRel.append(abs(w))
                    weightsByLayer.append(abs(w)/tensor_max)
                    weightsByLayerNonRel.append(abs(w))
                    it.iternext()
            if len(weightsByLayer)>0:
                allWeightsByLayer[layer.name] = np.array(weightsByLayer)
                allWeightsByLayerNonRel[layer.name] = np.array(weightsByLayerNonRel)
    return np.array(allWeights), allWeightsByLayer, np.array(allWeightsNonRel), allWeightsByLayerNonRel

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--model'   ,action='store',type='string',dest='inputModel'   ,default='train_simple/KERAS_check_best_model.h5', help='input model')
    parser.add_option('--relative-weight-max'   ,action='store',type='float',dest='relative_weight_max'   ,default=None, help='max relative weight')
    parser.add_option('--relative-weight-percentile'   ,action='store',type='float',dest='relative_weight_percentile'   ,default=None, help='relative weight percentile')
    parser.add_option('-o','--outputModel'   ,action='store',type='string',dest='outputModel'   ,default='prune_simple/pruned_model.h5', help='output directory')
    (options,args) = parser.parse_args()

    from models import three_layer_model
    from keras.layers import Input
    model = load_model(options.inputModel, custom_objects={'ZeroSomeWeights':ZeroSomeWeights})
    model.load_weights(options.inputModel)

    weightsPerLayer = {}
    droppedPerLayer = {}
    binaryTensorPerLayer = {}
    allWeightsArray,allWeightsByLayer,allWeightsArrayNonRel,allWeightsByLayerNonRel = getWeightArray(model)
    if options.relative_weight_percentile is not None:
        relative_weight_max = np.percentile(allWeightsArray,options.relative_weight_percentile,axis=-1)
    elif options.relative_weight_max is not None:
        relative_weight_max = options.relative_weight_max
    else:
        print('Need to set pruning criteria')
        sys.exit()
        
    for layer in model.layers:     
        droppedPerLayer[layer.name] = []
        if layer.__class__.__name__ in ['Dense', 'Conv1D', 'LSTM']:
            original_w = layer.get_weights()
            weightsPerLayer[layer.name] = original_w
            for my_weights in original_w:
                if len(my_weights.shape) < 2: # bias term, skip for now
                    continue
                #l1norm = tf.norm(my_weights,ord=1)
                elif len(my_weights.shape) == 2: # Dense
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                elif len(my_weights.shape) == 3: # Conv1D
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_0 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_reduce_max_0,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                with tf.Session():
                    #l1norm_val = float(l1norm.eval())
                    tensor_max = float(tensor_reduce_max_2.eval())
                it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readwrite'])                
                binaryTensorPerLayer[layer.name] = np.ones(my_weights.shape)
                while not it.finished:
                    w = it[0]
                    if abs(w)/tensor_max < relative_weight_max:
                        #print("small relative weight %e/%e = %e -> 0"%(abs(w), tensor_max, abs(w)/tensor_max))
                        w[...] = 0
                        droppedPerLayer[layer.name].append((it.multi_index, abs(w)))
                        binaryTensorPerLayer[layer.name][it.multi_index] = 0
                    it.iternext()
            #print('%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name,layer.count_params()))
            #converted_w = convert_kernel(original_w)
            converted_w = original_w
            layer.set_weights(converted_w)


    print('Summary:')
    totalDropped = sum([len(droppedPerLayer[layer.name]) for layer in model.layers])
    for layer in model.layers:
        print('%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name, layer.count_params()))
    print('%i total weights dropped out of %i total weights'%(totalDropped,model.count_params()))
    print('%.1f%% compression'%(100.*totalDropped/model.count_params()))
    model.save(options.outputModel)
    model.save_weights(options.outputModel.replace('.h5','_weights.h5'))
    print_model_to_json(model, options.outputModel.replace('.h5','.json'))
    
    # save binary tensor in h5 file 
    h5f = h5py.File(options.outputModel.replace('.h5','_drop_weights.h5'),'w')
    for layer, binary_tensor in binaryTensorPerLayer.items():
        h5f.create_dataset('%s'%layer, data = binaryTensorPerLayer[layer])
    h5f.close()

    # plot the distribution of weights
    if options.relative_weight_percentile is not None:
        your_percentile = options.relative_weight_percentile
    else:
        your_percentile = stats.percentileofscore(allWeightsArray, relative_weight_max)
    #percentiles = [5,16,50,84,95,your_percentile]
    percentiles = [5,95,your_percentile]
    #colors = ['r','r','r','r','r','g']
    colors = ['r','r','g']
    vlines = np.percentile(allWeightsArray,percentiles,axis=-1)
    xmin = np.amin(allWeightsArray[np.nonzero(allWeightsArray)])
    xmax = np.amax(allWeightsArray)
    xmin = 6e-8
    xmax = 1
    bins = np.linspace(xmin, xmax, 50)
    logbins = np.geomspace(xmin, xmax, 50)

    labels = []
    histos = []
    for key in reversed(sorted(allWeightsByLayer.keys())):
        labels.append(key)
        histos.append(allWeightsByLayer[key])        
    
    plt.figure()
    #plt.hist(allWeightsArray,bins=bins)
    #plt.hist(allWeightsByLayer.values(),bins=bins,histtype='bar',stacked=True,label=allWeightsByLayer.keys())
    plt.hist(histos,bins=bins,histtype='step',stacked=False,label=labels)
    plt.legend(prop={'size':10}, frameon=False)
    axis = plt.gca()
    ymin, ymax = axis.get_ylim()
    for vline, percentile, color in zip(vlines, percentiles, colors):
        if percentile==0: continue
        if vline < xmin: continue
        plt.axvline(vline, 0, 1, color=color, linestyle='dashed', linewidth=1, label = '%s%%'%percentile)
        plt.text(vline, ymax+0.01*(ymax-ymin), '%s%%'%percentile, color=color, horizontalalignment='center')
    plt.ylabel('Number of Weights')
    plt.xlabel('Absolute Relative Weights')
    plt.savefig(options.outputModel.replace('.h5','_weight_histogram.pdf'))

        
    plt.figure()
    #plt.hist(allWeightsArray,bins=logbins)
    #plt.hist(allWeightsByLayer.values(),bins=logbins,histtype='bar',stacked=True,label=allWeightsByLayer.keys())
    plt.hist(histos,bins=logbins,histtype='step',stacked=False,label=labels)
    plt.semilogx()
    plt.legend(prop={'size':10}, frameon=False)
    axis = plt.gca()
    ymin, ymax = axis.get_ylim()
    
    for vline, percentile, color in zip(vlines, percentiles, colors):
        if percentile==0: continue
        if vline < xmin: continue
        xAdd = 0
        yAdd = 0
        #if plotPercentile5 and percentile==84:
        #    xAdd=0.2
        #if plotPercentile16 and percentile==95:
        #    xAdd=1.2
        plt.axvline(vline, 0, 1, color=color, linestyle='dashed', linewidth=1, label = '%s%%'%percentile)
        plt.text(vline+xAdd, ymax+0.01*(ymax-ymin)+yAdd, '%s%%'%percentile, color=color, horizontalalignment='center')
    plt.ylabel('Number of Weights')
    plt.xlabel('Absolute Relative Weights')
    plt.figtext(0.25, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig(options.outputModel.replace('.h5','_weight_histogram_logx.pdf'))


    labels = []
    histos = []
    for key in reversed(sorted(allWeightsByLayerNonRel.keys())):
        labels.append(key)
        histos.append(allWeightsByLayerNonRel[key])
        
    xmin = np.amin(allWeightsArrayNonRel[np.nonzero(allWeightsArrayNonRel)])
    xmax = np.amax(allWeightsArrayNonRel)
    #bins = np.linspace(xmin, xmax, 100)
    bins = np.geomspace(xmin, xmax, 50)
    plt.figure()
    #plt.hist(allWeightsArrayNonRel,bins=bins)
    #plt.hist(allWeightsByLayerNonRel.values(),bins=bins,histtype='bar',stacked=True,label=allWeightsByLayer.keys())
    plt.hist(histos,bins=bins,histtype='step',stacked=False,label=labels)
    plt.semilogx(basex=2)
    plt.legend(prop={'size':10}, frameon=False, loc='upper left')
    plt.ylabel('Number of Weights')
    plt.xlabel('Absolute Value of Weights')
    plt.figtext(0.25, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig(options.outputModel.replace('.h5','_weight_nonrel_histogram_logx.pdf'))
