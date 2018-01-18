from keras.constraints import *
from keras.layers.merge import multiply

class ZeroSomeWeights(Constraint):
    """ZeroSomeWeights weight constraint.

    Constrains certain weights incident to each hidden unit
    to be zero.

    # Arguments
        binary_tensor: binary tensor of 0 or 1s corresponding to which weights to zero.
    """

    def __init__(self, binary_tensor=None):
        self.binary_tensor = binary_tensor

    def __call__(self, w):
        if self.binary_tensor is not None:
            w = multiply([w,  K.variable(value=self.binary_tensor)])
        return w

    def get_config(self):
        return {'binary_tensor': self.binary_tensor}

# Aliases.

zero_some_weights = ZeroSomeWeights
