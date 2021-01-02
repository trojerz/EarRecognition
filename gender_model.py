import math
import numpy as np
import matplotlib.pyplot as plt

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import Adam, Adamax, Adadelta, Nadam, SGD
from keras.callbacks import History

def train(data, batch_size, epochs, pool_size, kernel_size, n_classes):
    input_shape = (200, 200, 1) # BW pic
    [(X_train, Y_train), (X_valid, Y_valid),(X_test, Y_test)] = data
    X_train = X_train[:,:,:,np.newaxis] 
    X_valid = X_valid[:,:,:,np.newaxis]
    X_test = X_test[:,:,:,np.newaxis]
    # create model
    model = Sequential()
    model.add(Conv2D(16, kernel_size, padding = 'same', input_shape = input_shape, strides=1))
    model.add(Activation('relu'))  
    model.add(AveragePooling2D(pool_size, strides=2))
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('softmax')) 

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    adamax = Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

    nadam = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    sgd =SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

    optimizer = nadam

    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  

    history = History()
   
    #class_weight = {0:1, 1:5}

    history  = model.fit(X_train, Y_train, batch_size, epochs, verbose=1, validation_data=(X_valid, Y_valid), callbacks=[history], class_weight = class_weight)
    score = model.evaluate(X_test, Y_test, verbose=0)

    plt.figure(figsize=(20,12))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right', prop={'size': 24})
    plt.savefig('pic.png')

    print('Test accuracy:', score[1])
    return model