import os
import time
import sys
sys.path.append('..')

import cv2
from joblib import load
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import Sequential
from keras.models import load_model

# from keras.applications import InceptionV3
# from keras.applications.inception_v3 import preprocess_input

from keras.applications import Xception
from keras.applications.xception import preprocess_input

from keras.backend import resize_images
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten

# from image_preprocess import ImageProcessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Building CNN model from scratch
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

    model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy')

    return model


# Uses pre-trained Inception V3 image classifier model and apply transfer learning
def transfer_learning_model(image_shape, num_classes):
    model = Sequential()
    model.add(Xception(include_top=False, weights='imagenet', input_shape=image_shape, pooling='avg'))
    # model.add(Dense(8192, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))
    model.layers[0].trainable = False
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
    return model


def model_train(model, model_name, training_input=None, training_output=None, batch_size=64, epochs=100000, verbose=1,
                validation_generator=None, generator=None, use_multiprocessing=False, max_workers=1,
                max_queue_size=5, save_weights_only=True, validation_steps=None):
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1, save_weights_only=save_weights_only)
    early_stop = EarlyStopping(min_delta=0.01, patience=20, verbose=1, mode='min')
    if validation_generator is None:
        save_checkpoint.monitor = 'loss'
        early_stop.monitor = 'loss'
    else:
        validation_steps = 1
    callbacks = [save_checkpoint, early_stop]
    if generator:
        model.fit_generator(generator=generator, steps_per_epoch=(generator.n // batch_size), epochs=epochs,
                            verbose=verbose, callbacks=callbacks, validation_data=validation_generator,
                            use_multiprocessing=use_multiprocessing, workers=max_workers,
                            max_queue_size=max_queue_size, validation_steps=validation_steps)
    elif training_input and training_output:
        model.fit(training_input, training_output, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True,
                  callbacks=callbacks)


def predict_breed(model, image, default_size=(250, 250),
                  ordered_classes_dict=load(os.path.join(os.path.dirname(__file__),
                                                         'telegram_bot', 'index_to_doggo.pkl'))):
    resized_image = cv2.resize(image, default_size)
    ndarray_image = np.reshape(resized_image, ((1, ) + resized_image.shape))
    casted_image = ndarray_image.astype(np.float32)
    preprocessed_image = preprocess_input(casted_image)
    breed_pred = model.predict(preprocessed_image)
    breed_pred = breed_pred.reshape(breed_pred.shape[1])

    # breed = ordered_classes_list[np.argmax(breed_pred, axis=1)[0]]
    # breed_prob = np.max(breed_pred, axis=1)[0]

    top_3_breeds_pred = np.argsort(breed_pred)[-3:][::-1]
    breeds = [ordered_classes_dict[breed_index] for breed_index in top_3_breeds_pred]
    # breeds = [ordered_classes_list[index] for index in top_3_breeds_pred]
    breeds_prob = breed_pred[top_3_breeds_pred]

    return [breeds, breeds_prob]


def decode_map(ndarray, mapping_list):
    max_indices = ndarray.argmax(axis=1).tolist()
    mapped_values = [mapping_list[index] for index in max_indices]
    return mapped_values


# def preprocess_dir_images(image_directory, image_size, classes_list):
#     image_array = []
#     image_labels_array = []
#     placeholder_img = np.zeros(image_size, np.uint8)
#     image_classes = os.listdir(image_directory)
#     for image_class in image_classes:
#         image_class_dir = os.path.join(image_directory, image_class)
#         image_filenames = os.listdir(image_class_dir)
#         for image_filename in image_filenames:
#             image_filepath = os.path.join(image_class_dir, image_filename)
#             resized_image = cv2.resize(preprocess_input(cv2.imread(image_filepath)),
#                                        dst=placeholder_img, dsize=image_size[0:2])
#             image_array.append(resized_image)
#             image_label_array = np.zeros(len(classes_list))
#             label_index = classes_list.index(image_class)
#             image_label_array[label_index] = 1
#             image_labels_array.append(image_label_array)
#     return np.asarray(image_array), np.asarray(image_labels_array)


if __name__ == '__main__':
    save_weights_name = 'models/doggo_classifier_weights_v2.h5'
    save_model_name = 'models/doggo_classifier_model_v1.h5'
    batch_size = 200
    image_shape = (250, 250, 3)
    number_of_classes = len(os.listdir('data/train'))

    # # Custom image pre-processing
    # processor = ImageProcessor('images', '../image_scrapper/breeds.csv', save_encoding_fname='y_encoding.pkl')
    # x, y = processor.load_images(pad=True)
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, random_state=15, stratify=np.unique(y))

    # # Training image data generator with Keras pre-processing function
    # train_datagen = ImageDataGenerator(rotation_range=45, horizontal_flip=True, data_format='channels_last',
    #                                    zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
    #                                    fill_mode='nearest', shear_range=0.2, preprocessing_function=preprocess_input)
    # train_generator = train_datagen.flow_from_directory('data/train', target_size=image_shape[:-1],
    #                                                     batch_size=batch_size, class_mode='categorical')
    #
    # val_datagen = ImageDataGenerator(data_format='channels_last', fill_mode='nearest',
    #                                  preprocessing_function=preprocess_input, rotation_range=30,
    #                                  horizontal_flip=True, zoom_range=0.2, width_shift_range=0.2,
    #                                  height_shift_range=0.2)
    # val_generator = val_datagen.flow_from_directory('data/test', target_size=image_shape[:-1],
    #                                                 batch_size=batch_size, class_mode='categorical')

    # Building of transfer learning model, loading weights and training using train image data generator
    doggo_model = transfer_learning_model(image_shape=image_shape, num_classes=number_of_classes)
    doggo_model.load_weights(save_weights_name)
    # model_train(doggo_model, save_weights_name, generator=train_generator, max_queue_size=True, max_workers=-1,
    #             save_weights_only=True, validation_generator=val_generator)

    # # Model testing
    # val_datagen = ImageDataGenerator(data_format='channels_last', fill_mode='nearest',
    #                                  preprocessing_function=preprocess_input)
    # val_generator = val_datagen.flow_from_directory('data/test', target_size=image_shape[:-1], batch_size=100,
    #                                                 shuffle=False)
    # breed_preds = doggo_model.predict_generator(val_generator)
    # decoded_preds = decode_map(breed_preds, os.listdir(r'data\train'))

    # # Testing on single image prediction
    # doggo_model = load_model(save_model_name)
    print(time.ctime(), 'start')
    doggo_friends_dict = {}
    friends_folder = 'data/friends'
    for doggo_breed in os.listdir(friends_folder):
        breed_folder = os.path.join(friends_folder, doggo_breed)
        doggo_friends_dict[doggo_breed] = {}
        for breed_image in os.listdir(breed_folder):
            breed_image_filepath = os.path.join(breed_folder, breed_image)
            breed_pred = predict_breed(doggo_model, cv2.imread(breed_image_filepath))
            doggo_friends_dict[doggo_breed][breed_image] = breed_pred
    print(time.ctime(), 'end')

    # model.evaluate_generator(generator=val_generator)
