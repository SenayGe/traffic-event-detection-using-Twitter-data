from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D
# define models
def mlp_model(input_size):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model (feture_size):
    num_outputs = 1
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=( 1,feture_size), padding = 'same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding = 'same'))

    model.add(MaxPooling1D(pool_size=2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_outputs, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model