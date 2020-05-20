import numpy as np
from generator import gen_data
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Lambda, Activation, concatenate, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import LearningRateScheduler, EarlyStopping
from functools import partial, update_wrapper
from itertools import chain
from random import randint
import activations, losses
import os
import sherpa
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import h5py

client = sherpa.Client()
trial = client.get_trial()


# Sets up the convolutional branch 
inputs = Input(shape=(72, 60, 2))

x = Conv2D(64, (3, 3))(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = GlobalMaxPooling2D()(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dense(256)(x)
x = Activation('relu')(x)

cnn = Model(inputs, x)

# Sets up the scalar valued ANN portion 
# Cwave boolean is simply whether or not to include original cwave parameters. 
if trial.parameters['cwave']:
    inp = Input(shape=(32, ))  # 'hsSM', 'hsWW3v2', 'hsALT', 'altID', 'target' -> dropped
else:
    inp = Input(shape=(12, ))
x = Dense(units=256, activation='relu')(inp)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)



ann = Model(inputs=inp, outputs=x)

combinedInput = concatenate([cnn.output, ann.output])
x = Dense(256, activation="relu")(combinedInput)
x = Dense(128, activation="relu")(x)
x = Dropout(0.33716497130943546)(x)
x = Dense(1, activation="softplus")(x)


model = Model(inputs=[cnn.input, ann.input], outputs=x)

# Uses a set decay schedule without validation monitoring 
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.40, step_size=4):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        if epoch >= 10 and epoch < 20:
            exponent = 1
        elif epoch >= 20 and epoch <= 118:
            exponent = 2
        else:
            exponent = 3
        return initial_lr * (decay_factor ** exponent)
    
    return LearningRateScheduler(schedule)

reduce_lr = step_decay_schedule(initial_lr=0.00025897101528140915, decay_factor=0.40, step_size=4)

opt = Adam(lr=0.00025897101528140915)

send_metrics_cbk = client.keras_send_metrics(trial, objective_name='val_loss', context_names=['loss', 'val_loss'])

model.compile(loss='mean_squared_error',
              optimizer=opt)

file_path = ''
dataset = h5py.File(file_path, 'r')

batch_size = 128
epochs = 30
file = file_path 
steps = int(dataset["X_train"].shape[0] * trial.parameters['size']) // batch_size + 1

if trial.parameters['cwave']:
    history = model.fit_generator(gen_data(file, 'train', batch_size),
                                  steps_per_epoch=steps, 
                                  epochs=epochs,
                                  validation_data = gen_data(file, 'test', batch_size), 
                                  validation_steps=dataset["X_test"].shape[0]//batch_size + 1,
                                  callbacks=[reduce_lr, send_metrics_cbk],
                                  verbose = 1)
else:
    history = model.fit_generator(gen_data(file, 'train', batch_size, list(range(11, 31))),
                                  steps_per_epoch=steps, 
                                  epochs=epochs,
                                  validation_data = gen_data(file, 'test', batch_size, list(range(11, 31))), 
                                  validation_steps=dataset["X_test"].shape[0]//batch_size + 1,
                                  callbacks=[reduce_lr, send_metrics_cbk],
                                  verbose = 1)

    # Saves the model
save_path = ''
model.save(save_path + 'ensemble_{}.h5'.format(trial.id))
