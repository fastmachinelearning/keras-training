from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, Conv1D, Input, SpatialDropout1D, GRU, MaxPooling1D, AveragePooling1D, SimpleRNN, LSTM, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.regularizers import l1
import h5py
from constraints import *
from quantized_layers import BinaryDense, TernaryDense, QuantizedDense
from quantized_ops import binary_tanh as binary_tanh_op
from quantized_ops import ternarize
from quantized_ops import quantized_relu as quantize_op

def binary_tanh(x):
    return binary_tanh_op(x)

def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

def quantized_relu(x):
    return quantize_op(x,nb=4)
    
def dense_model(Inputs, nclasses, l1Reg=0, dropoutRate=0.25):
    """
    Dense matrix, defaults similar to 2016 DeepCSV training
    """
    x = Dense(200, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu', kernel_initializer='lecun_uniform', name='fc2_relu', W_regularizer=l1(l1Reg))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu', kernel_initializer='lecun_uniform', name='fc3_relu', W_regularizer=l1(l1Reg))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu', kernel_initializer='lecun_uniform', name='fc4_relu', W_regularizer=l1(l1Reg))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu', kernel_initializer='lecun_uniform', name='fc5_relu', W_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name = 'output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def two_layer_model(Inputs, nclasses, l1Reg=0):
    """
    One hidden layer model
    """
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    predictions = Dense(nclasses, activation='sigmoid', kernel_initializer='lecun_uniform', 
                        name = 'output_sigmoid', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def two_layer_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    One hidden layer model
    """
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_relu'][()].tolist()))(Inputs)
    predictions = Dense(nclasses, activation='sigmoid', kernel_initializer='lecun_uniform', 
                        name = 'output_sigmoid', W_regularizer=l1(l1Reg), 
                        kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc2_relu', W_regularizer=l1(l1Reg))(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc3_relu', W_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model_batch_norm(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1')(x)
    x = Activation(activation='relu', name='relu1')(x)
              
    x = Dense(32, kernel_initializer='lecun_uniform', 
              name='fc2_relu', W_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2')(x)
    x = Activation(activation='relu', name='relu2')(x)
    
    x = Dense(32, kernel_initializer='lecun_uniform', 
              name='fc3_relu', W_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3')(x)
    x = Activation(activation='relu', name='relu3')(x)
    
    x = Dense(nclasses, kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn4')(x)
    predictions = Activation(activation='softmax', name='softmax')(x)

    model = Model(inputs=Inputs, outputs=predictions)
    return model
    
def three_layer_model_binary(Inputs, nclasses, l1Reg=0):
    """
    Three hidden layers model
    """
     
    model = Sequential()
    
    model.add(BinaryDense(64, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc1', input_shape=(16,)))
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1'))
    model.add(Activation(binary_tanh, name='act{}'.format(1)))
    
    model.add(BinaryDense(32, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc2'))  
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2'))
    model.add(Activation(binary_tanh, name='act{}'.format(2)))  
    
    model.add(BinaryDense(32, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc3'))   
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3'))
    model.add(Activation(binary_tanh, name='act{}'.format(3)))  
        
    model.add(BinaryDense(nclasses, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='output'))
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn'))
    
    return model                                                       

def three_layer_model_ternary(Inputs, nclasses, l1Reg=0):
    """
    Three hidden layers model
    """
     
    model = Sequential()
    
    model.add(TernaryDense(64, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc1', input_shape=(16,)))
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1'))
    model.add(Activation(ternary_tanh, name='act{}'.format(1)))     
    
    model.add(TernaryDense(32, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc2'))  
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2'))
    model.add(Activation(ternary_tanh, name='act{}'.format(2)))  
    
    model.add(TernaryDense(32, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc3'))   
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3'))
    model.add(Activation(ternary_tanh, name='act{}'.format(3)))      
    
    model.add(TernaryDense(nclasses, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='output'))
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn'))
    
    return model 

def three_layer_model_qnn(Inputs, nclasses, l1Reg=0):
    """
    Three hidden layers model
    """
     
    model = Sequential()
    model.add(QuantizedDense(64, nb=4, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc1', input_shape=(16,)))
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1'))
    model.add(Activation(quantized_relu, name='act{}'.format(1)))     
    
    model.add(QuantizedDense(32, nb=4, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc2'))  
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2'))
    model.add(Activation(quantized_relu, name='act{}'.format(2)))  
    
    model.add(QuantizedDense(32, nb=4, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='fc3'))   
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3'))
    model.add(Activation(quantized_relu, name='act{}'.format(3)))      
    
    model.add(QuantizedDense(nclasses, nb=4, H='Glorot', kernel_lr_multiplier='Glorot', use_bias=False, name='output'))
    model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn'))
    
    return model 
    
def three_layer_model_tanh(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, activation='tanh', kernel_initializer='lecun_uniform', 
              name='fc1_tanh', W_regularizer=l1(l1Reg))(Inputs)
    x = Dense(32, activation='tanh', kernel_initializer='lecun_uniform', 
              name='fc2_tanh', W_regularizer=l1(l1Reg))(x)
    x = Dense(32, activation='tanh', kernel_initializer='lecun_uniform', 
              name='fc3_tanh', W_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    Two hidden layers model
    """
    h5f = h5py.File(h5fName)
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_relu'][()].tolist()))(Inputs)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc2_relu', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc2_relu'][()].tolist()))(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc3_relu', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc3_relu'][()].tolist()))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg), 
                        kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def three_layer_model_tanh_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    Two hidden layers model
    """
    h5f = h5py.File(h5fName)
    x = Dense(64, activation='tanh', kernel_initializer='lecun_uniform', 
              name='fc1_tanh', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_tanh'][()].tolist()))(Inputs)
    x = Dense(32, activation='tanh', kernel_initializer='lecun_uniform', 
              name='fc2_tanh', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc2_tanh'][()].tolist()))(x)
    x = Dense(32, activation='tanh', kernel_initializer='lecun_uniform', 
              name='fc3_tanh', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc3_tanh'][()].tolist()))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg), 
                        kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
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
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(Inputs)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(x)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def conv1d_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    Conv1D model, kernel size 40
    """
    h5f = h5py.File(h5fName)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), 
               kernel_constraint = zero_some_weights(binary_tensor=h5f['conv1_relu'][()].tolist()))(Inputs)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), 
               kernel_constraint = zero_some_weights(binary_tensor=h5f['conv2_relu'][()].tolist()))(x)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), 
               kernel_constraint = zero_some_weights(binary_tensor=h5f['conv3_relu'][()].tolist()))(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_relu'][()].tolist()))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg), 
                        kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def conv1d_small_model(Inputs, nclasses, l1Reg=0):
    """
    Conv1D small model, kernel size 4
    """
    x = Conv1D(filters=3, kernel_size=4, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(Inputs)
    x = Conv1D(filters=2, kernel_size=4, strides=2, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(x)
    x = Conv1D(filters=1, kernel_size=4, strides=3, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu', W_regularizer=l1(l1Reg))(x)
    x = Flatten()(x)
    x = Dense(5, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model


def conv1d_small_model_constraint(Inputs, nclasses, l1Reg=0, h5fName=None):
    """
    Conv1D small model, kernel size 4
    """
    h5f = h5py.File(h5fName)
    x = Conv1D(filters=3, kernel_size=4, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), 
               kernel_constraint = zero_some_weights(binary_tensor=h5f['conv1_relu'][()].tolist()))(Inputs)
    x = Conv1D(filters=2, kernel_size=4, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), 
               kernel_constraint = zero_some_weights(binary_tensor=h5f['conv2_relu'][()].tolist()))(x)
    x = Conv1D(filters=1, kernel_size=4, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu', W_regularizer=l1(l1Reg), 
               kernel_constraint = zero_some_weights(binary_tensor=h5f['conv3_relu'][()].tolist()))(x)
    x = Flatten()(x)
    x = Dense(5, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', W_regularizer=l1(l1Reg), 
              kernel_constraint = zero_some_weights(binary_tensor=h5f['fc1_relu'][()].tolist()))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', W_regularizer=l1(l1Reg), 
                        kernel_constraint = zero_some_weights(binary_tensor=h5f['output_softmax'][()].tolist()))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def conv2d_model(Inputs, nclasses, l1Reg=0):
    """
    Conv2D model, kernel size (11,11), (3,3), (3,3)
    """
    x = Conv2D(filters=8, kernel_size=(11,11), strides=(1,1), padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu')(Inputs)
    x = Conv2D(filters=4, kernel_size=(3,3), strides=(2,2), padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu')(x)
    x = Conv2D(filters=2, kernel_size=(3,3), strides=(2,2), padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def rnn_model(Inputs, nclasses, l1Reg=0):
    """
    Simple RNN model
    """
    x = SimpleRNN(72,return_sequences=True)(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_densef')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def lstm_model(Inputs, nclasses, l1Reg=0,l1RegR=0):
    """
    Basic LSTM model
    """
    x = LSTM(16,return_sequences=False,  kernel_regularizer=l1(l1Reg),recurrent_regularizer=l1(l1RegR),activation='relu',kernel_initializer='lecun_uniform',name='lstm_lstm')(Inputs)
    #x = Flatten()(x)
    x = Dropout(0.1)(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_densef')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def lstm_model_constraint(Inputs, nclasses, l1Reg=0,l1RegR=0,h5fName=None):
    """
    Basic LSTM model
    """
    h5f = h5py.File(h5fName)
    x = LSTM(16,return_sequences=False,kernel_regularizer=l1(l1Reg),recurrent_regularizer=l1(l1RegR),name='lstm_lstm',recurrent_constraint = zero_some_weights(binary_tensor=h5f['lstm_lstm'][()].tolist()))(Inputs)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', kernel_constraint = zero_some_weights(binary_tensor=h5f['rnn_densef'][()].tolist()), name='rnn_densef')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def lstm_model_full(Inputs, nclasses, l1Reg=0):
    """
    LSTM model akin to what Sid is using for his studies
    """
    x = Conv1D(32, 2, activation='relu', name='particles_conv0', kernel_initializer='lecun_uniform', padding='same')(Inputs)
    x = Conv1D(16, 4, activation='relu', name='particles_conv1', kernel_initializer='lecun_uniform', padding='same')(x)
    x = LSTM(72,return_sequences=True)(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_dense2')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_dense3')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_densef')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def gru_model(Inputs, nclasses, l1Reg=0,l1RegR=0):
    """                                                                                                                                                                                                                                                                         
    Basic GRU model                                                                                                                                                                                                                                                             
    """
    x = GRU(20,kernel_regularizer=l1(l1Reg),recurrent_regularizer=l1(l1RegR),activation='relu', recurrent_activation='sigmoid', name='gru_selu',)(Inputs)
    #x = GRU(20,kernel_regularizer=l1(l1Reg),recurrent_regularizer=l1(l1RegR),activation='selu', recurrent_activation='hard_sigmoid', name='gru_selu',)(Inputs)                                                                                                                
    x = Dense(20,kernel_regularizer=l1(l1Reg),activation='relu', kernel_initializer='lecun_uniform', name='dense_relu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_densef')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def gru_model_constraint(Inputs, nclasses, l1Reg=0,l1RegR=0,h5fName=None):
    """                                                                                                                                                                                                                                                                         
    Basic GRU  model                                                                                                                                                                                                                                                            
    """
    h5f = h5py.File(h5fName)
    x = GRU(20,kernel_regularizer=l1(l1Reg),recurrent_regularizer=l1(l1RegR),activation='selu',recurrent_activation='hard_sigmoid',name='gru_selu',recurrent_constraint = zero_some_weights(binary_tensor=h5f['gru_selu'][()].tolist()))(Inputs)
    x = Dense(20,kernel_regularizer=l1(l1Reg),activation='relu', kernel_initializer='lecun_uniform',kernel_constraint = zero_some_weights(binary_tensor=h5f['dense_relu'][()].tolist()), name='dense_relu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', kernel_constraint = zero_some_weights(binary_tensor=h5f['rnn_densef'][()].tolist()), name='rnn_densef')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model


if __name__ == '__main__':
    print(conv1d_model(Input(shape=(100,10,)), 2).summary())
    
    print(conv2d_model(Input(shape=(10,10,3,)), 2).summary())
