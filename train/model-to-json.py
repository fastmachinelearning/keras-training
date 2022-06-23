import sys
import os
import keras
from keras.models import load_model
from optparse import OptionParser
from train import print_model_to_json
from constraints import ZeroSomeWeights
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "-m",
        "--model",
        action="store",
        type="string",
        dest="inputModel",
        default="train_simple/KERAS_check_best_model.h5",
        help="input model",
    )
    (options, args) = parser.parse_args()

    model = load_model(options.inputModel)
    print_model_to_json(model, options.inputModel.replace(".h5", ".json"))
