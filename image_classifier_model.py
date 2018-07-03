import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from keras import Sequential
from keras.models import load_model
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, metrics, losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten

from image_preprocess import ImageProcessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model_builder(model_input, model_output):
    model = Sequential()

    # 2*[Convolution operation > Nonlinear activation (relu)] > Pooling operation
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=tuple(model_input.shape[1:])))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(model_output.shape[1], activation='softmax'))

    # Nesterov momentum included for parameters update, makes correction to parameters update values
    # by taking into account the approximated future value of the objective function
    # However does not account for the importance for each parameter when performing updates
    # op = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Nesterov Momentum Adaptive Moment Estimation
    # op = optimizers.Nadam()

    # RMSProp
    op = optimizers.RMSprop()

    # Adadelta optimiser algorithm
    # op = optimizers.Adadelta()

    model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy')

    return model


def transfer_learning_model(image_shape, num_classes):
    model = Sequential()
    model.add(InceptionV3(include_top=False, weights='imagenet', input_shape=image_shape,
                          pooling='avg'))
    model.add(Dense(num_classes, activation='softmax'))
    model.layers[0].trainable = False
    op = optimizers.RMSprop()
    model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy')
    return model


def model_train(model, model_name, training_input=None, training_output=None, batch_size=64, epochs=100000, verbose=1,
                test_input=None, test_output=None, generator=None, use_multiprocessing=False, max_workers=1,
                max_queue_size=5):
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1, save_weights_only=True)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    if test_input is None:
        validation_data = None
        save_checkpoint.monitor = 'loss'
        early_stop.monitor = 'loss'
    else:
        validation_data = (test_input, test_output)
    callbacks = [save_checkpoint, early_stop]
    if generator:
        model.fit_generator(generator=generator, steps_per_epoch=(generator.n//batch_size), epochs=epochs,
                            verbose=verbose, callbacks=callbacks, validation_data=validation_data,
                            use_multiprocessing=use_multiprocessing, workers=max_workers,
                            max_queue_size=max_queue_size)
    elif training_input and training_output:
        model.fit(training_input, training_output, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True,
                  callbacks=callbacks, validation_data=validation_data)


if __name__ == '__main__':

    # processor = ImageProcessor('images', '../image_scrapper/breeds.csv', save_encoding_fname='y_encoding.pkl')
    # x, y = processor.load_images(pad=True)
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, random_state=15, stratify=np.unique(y))

    name = 'doggo_classifier.h5'
    batch_size = 64
    image_shape = (250, 250, 3)
    number_of_classes = len(os.listdir('data/train'))

    datagen = ImageDataGenerator(rotation_range=45, horizontal_flip=True, data_format='channels_last', zoom_range=0.2,
                                 width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', shear_range=0.2,
                                 preprocessing_function=preprocess_input)

    train_generator = datagen.flow_from_directory(r'data\train', target_size=image_shape[:-1], batch_size=batch_size,
                                                  class_mode='categorical')

    doggo_model = transfer_learning_model(image_shape=image_shape, num_classes=number_of_classes)
    doggo_model.load_weights(name)

    model_train(doggo_model, name, generator=train_generator, max_queue_size=True, max_workers=4)

    print('training_completed')
