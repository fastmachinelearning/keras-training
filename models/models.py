from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, Conv1D, Input, SpatialDropout1D, GRU, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.regularizers import l1
import h5py
from constraints import *

def dense_model(Inputs, nclasses, l1Reg=0, dropoutRate=0.25):
    """
    Dense matrix, defaults similar to 2016 DeepCSV training
    """
    x = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu')(Inputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='fc2_relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='fc3_relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='fc4_relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='fc5_relu')(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name = 'output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def two_layer_model(Inputs, nclasses, l1Reg=0):
    """
    One hidden layer model
    """
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    predictions = Dense(nclasses, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'output_sigmoid', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def two_layer_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    One hidden layer model
    """
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_relu'][()].tolist()))(Inputs)
    predictions = Dense(nclasses, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'output_sigmoid', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc2_relu', W_regularizer=l1(l1Reg))(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc3_relu', W_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    Two hidden layers model
    """
    h5f = h5py.File(h5fName)
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_relu'][()].tolist()))(Inputs)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc2_relu', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['fc2_relu'][()].tolist()))(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc3_relu', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['fc3_relu'][()].tolist()))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def linear_model(Inputs, nclasses, l1Reg=0):
    """
    Linear model
    """
    predictions = Dense(nclasses, activation='linear', kernel_initializer='lecun_uniform', name='output_linear')(Inputs)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def conv1d_model(Inputs, nclasses, l1Reg=0):
    """
    Conv1D model, kernel size 40
    """
    nConstituents = int(Inputs.shape[1])
    x = Conv1D(filters=32, kernel_size=int(nConstituents/2.5), strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(Inputs)
    #x = Dropout(rate=0.2)(x)    
    #x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(filters=32, kernel_size=int(nConstituents/2.5), strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(x)
    #x = Dropout(rate=0.2)(x)    
    #x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(filters=32, kernel_size=int(nConstituents/2.5), strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(x)
    #x = Dropout(rate=0.2)(x)    
    #x = GRU(int(nConstituents/2.),go_backwards=True,implementation=2,name='gru1_tanh')(x)
    #x = Dropout(rate=0.2)(x)    
    x = Flatten()(x)
    x = Dense(int(nConstituents/2.), activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu', W_regularizer=l1(l1Reg))(x)
    #x = Dropout(rate=0.2)(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print model.summary()
    return model

def conv1d_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    Conv1D model, kernel size 40
    """
    h5f = h5py.File(h5fName)
    x = Conv1D(filters=32, kernel_size=1, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=False, name='conv1_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), kernel_constraint = zero_some_weights(binary_tensor=h5f['conv1_relu'][()].tolist()))(Inputs)
    x = Flatten()(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)

    return model

def conv2d_model(Inputs, nclasses, l1Reg=0):
    """
    Conv2D model, kernel size (3,3)
    """
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
               kernel_initializer='he_normal', use_bias=False, name='conv2_relu',
               activation = 'relu')(Inputs)
    x = Flatten()(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)

    return model


if __name__ == '__main__':
    print conv1d_model(Input(shape=(100,10,)), 2).summary()
    
    print conv2d_model(Input(shape=(10,10,3,)), 2).summary()
