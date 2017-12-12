from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, Conv1D, Input
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

def conv1d_model(Inputs, nclasses):
    """
    Conv1D model, kernel size 1
    """
    x = Conv1D(filters=32, kernel_size=1, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=False, name='conv1_relu',
               activation = 'relu')(Inputs)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', name='output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)

    return model

def conv2d_model(Inputs, nclasses):
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
