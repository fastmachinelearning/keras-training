from keras.regularizers import *

class L1L2Lp(Regularizer):
    """Regularizer for L1, L2, and Lp regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
        lp: Float; Lp regularization factor.
        p: Float; Lp regularization exponent.
    """

    def __init__(self, l1=0., l2=0., lp=0., p=0.5):        
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.lp = K.cast_to_floatx(lp)
        self.p = K.cast_to_floatx(p)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        if self.lp:
            regularization += K.sum(self.lp * K.pow(K.abs(x),self.p))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'lp': float(self.lp), 
                'p': float(self.p)}


# Aliases.


def lp(l=0.01,p=0.5):
    return L1L2Lp(lp=l, p=p)
def l1l2lp(l1=0.01,l2=0.01,lp=0.01,p=0.5):
    return L1L2Lp(l1=l1,l2=l2,lp=lp, p=p)
