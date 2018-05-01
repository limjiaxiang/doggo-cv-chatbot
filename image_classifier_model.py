from keras import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten

from image_preprocess import ImageProcessor


def model_builder(model_input, model_output):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=tuple(model_input.shape[1:])))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(tuple(model_output.shape[1:]), activation='softmax'))

    op = optimizers.SGD(lr=0.001)

    model.compile(optimizer=op, metrics=['accuracy'],
                  loss='categorical_crossentropy')

    return model


def model_train(model, model_name, training_input, training_output):
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    try:
        model.fit(training_input, training_output, batch_size=100, epochs=100000, verbose=2,
                  validation_split=0.2, shuffle=True,)
                  # callbacks=[save_checkpoint, early_stop])
    except Exception as e:
        print(e)


if __name__ == '__main__':

    name = 'doggo_classifier.h5'

    processor = ImageProcessor('images', '../image_scrapper/breeds.csv')
    x, y = processor.load_images(padding=True)

    model_v1 = model_builder(x, y)

    model_train(model_v1, name, x, y)
