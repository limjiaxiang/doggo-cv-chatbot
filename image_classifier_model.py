import os

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, metrics, losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten

from image_preprocess import ImageProcessor


def model_builder(model_input, model_output):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=tuple(model_input.shape[1:])))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(model_output.shape[1], activation='softmax'))

    # Nesterov momentum included for parameters update, makes correction to parameters update values
    # by taking into account the approximated future value of the objective function
    # However does not account for the importance for each parameter when performing updates
    # op = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Switched to Nesterov Momentum Adaptive Moment Estimation with default parameters
    op = optimizers.Nadam()

    model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy')

    return model


def model_train(model, model_name, training_input=None, training_output=None, batch_size=64, epochs=100000, verbose=1,
                test_input=None, test_output=None, generator=None, pickle_safe=False, max_workers=1, max_q_size=10):
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    if test_input is None:
        validation_data = None
    else:
        validation_data = (test_input, test_output)
    if generator:
        model.fit_generator(generator=generator, steps_per_epoch=(generator.x.shape[0]//batch_size), epochs=epochs,
                            verbose=verbose, callbacks=[save_checkpoint, early_stop], validation_data=validation_data,
                            pickle_safe=pickle_safe, workers=max_workers,  max_q_size=max_q_size)
    elif training_input and training_output:
        model.fit(training_input, training_output, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True,
                  callbacks=[save_checkpoint, early_stop], validation_data=validation_data)


if __name__ == '__main__':

    name = 'doggo_classifier.h5'

    # processor = ImageProcessor('images', '../image_scrapper/breeds.csv')
    # x, y = processor.load_images(padding=True)

    # joblib.dump(x, 'x.pkl', compress=9)
    # joblib.dump(y, 'y.pkl', compress=9)

    x = joblib.load('x.pkl')
    y = joblib.load('y.pkl')

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=15)

    datagen = ImageDataGenerator(rotation_range=90, horizontal_flip=True, data_format='channels_last', zoom_range=0.5,
                                 width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', shear_range=0.2)

    bth_size = 32
    doggo_model = model_builder(train_x, train_y)
    model_train(doggo_model, name, generator=datagen.flow(train_x, train_y, batch_size=bth_size),
                test_input=test_x, test_output=test_y)
