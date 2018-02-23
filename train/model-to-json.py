import sys
import os
import keras
from keras.models import load_model
from optparse import OptionParser

def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'wb')
    jsonString = keras_model.to_json()
    print jsonString
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--model'   ,action='store',type='string',dest='inputModel'   ,default='train_simple/KERAS_check_best_model.h5', help='input model')
    (options,args) = parser.parse_args()

    model = load_model(options.inputModel)
    print_model_to_json(model,options.inputModel.replace('.h5','.json'))
