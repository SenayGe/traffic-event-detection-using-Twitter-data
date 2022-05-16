# define baseline model
def baseline_model(input):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=input, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model