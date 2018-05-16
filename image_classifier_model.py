import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras import Sequential
from keras.models import load_model
from keras.applications import InceptionV3, ResNet50, MobileNet, DenseNet121
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.resnet50 import preprocess_input
# from keras.applications.mobilenet import preprocess_input
# from keras.applications.densenet import preprocess_input
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


def transfer_learning_model(model_input, model_output):
    model = Sequential()
    model.add(InceptionV3(include_top=False, weights='imagenet', input_shape=tuple(model_input.shape[1:]),
                          pooling='avg'))
    model.add(Dense(model_output.shape[1], activation='softmax'))
    model.layers[0].trainable = False
    op = optimizers.RMSprop()
    model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy')
    return model


def model_train(model, model_name, training_input=None, training_output=None, batch_size=64, epochs=100000, verbose=1,
                test_input=None, test_output=None, generator=None, use_multiprocessing=False, max_workers=1,
                max_queue_size=5):
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    if test_input is None:
        validation_data = None
    else:
        validation_data = (test_input, test_output)
    if generator:
        model.fit_generator(generator=generator, steps_per_epoch=(generator.x.shape[0]//batch_size), epochs=epochs,
                            verbose=verbose, callbacks=[save_checkpoint, early_stop], validation_data=validation_data,
                            use_multiprocessing=use_multiprocessing, workers=max_workers,  max_queue_size=max_queue_size)
    elif training_input and training_output:
        model.fit(training_input, training_output, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True,
                  callbacks=[save_checkpoint, early_stop], validation_data=validation_data)


if __name__ == '__main__':

    name = 'doggo_classifier.h5'

    # processor = ImageProcessor('images', '../image_scrapper/breeds.csv', save_encoding_fname='y_encoding.pkl')
    # x, y = processor.load_images(pad=True)
    #
    # os.chdir('..')
    #
    # joblib.dump(x, 'x_dataset_normalise.pkl', compress=9)
    # joblib.dump(y, 'y.pkl', compress=9)

    x = joblib.load('x_dataset_normalise_stretch.pkl')
    y = joblib.load('y.pkl')

    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, random_state=15, stratify=np.unique(y))
    #
    # datagen = ImageDataGenerator(rotation_range=45, horizontal_flip=True, data_format='channels_last', zoom_range=0.2,
    #                              width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', shear_range=0.2)

    # bth_size = 64
    # doggo_model = model_builder(train_x, train_y)
    # doggo_model = load_model(name)
    # doggo_model = transfer_learning_model(train_x, train_y)
    # model_train(doggo_model, name, generator=datagen.flow(train_x, train_y, batch_size=bth_size),
    #             test_input=test_x, test_output=test_y, max_queue_size=True, max_workers=4)

    plt.subplot(1, 2, 1)
    plt.title(np.argmax(y[np.where(y[5] == 1)[0][0]]))
    plt.axis('off')
    plt.imshow(x[5].astype(np.uint8))

    plt.subplot(1, 2, 2)
    plt.title(np.argmax(y[np.where(y[7] == 1)[0][0]]))
    plt.axis('off')
    plt.imshow(x[7].astype(np.uint8))
    plt.show()

    print('hello')

    # InceptionV3(include_top=False, pooling='avg', )