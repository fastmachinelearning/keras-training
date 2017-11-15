from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D
from keras.models import Model

def dense_model(Inputs,nclasses,dropoutRate=0.25):
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

def two_layer_model(Inputs, nclasses):
    """
    One hidden layer model
    """
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu')(Inputs)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name = 'output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model(Inputs, nclasses):
    """
    Two hidden layers model
    """
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu')(Inputs)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc2_relu')(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc3_relu')(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def linear_model(Inputs, nclasses):
    """
    Linear model
    """
    predictions = Dense(nclasses, activation='linear', kernel_initializer='lecun_uniform', name='output_linear')(Inputs)
    model = Model(inputs=Inputs, outputs=predictions)
    return model
