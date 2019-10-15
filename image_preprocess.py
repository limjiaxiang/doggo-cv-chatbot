import os
import sys
import csv

import numpy as np
import cv2
from sklearn.externals import joblib


class ImageProcessor(object):
    def __init__(self, image_folder, classes_csv=None, classes_lst=None,
                 save_encoding_fname=None, load_encoding_fname=None):
        """
        Initialise image processor object, internal module used for preprocessing scraped/downloaded images
        """
        self.original_folder = os.getcwd()
        os.chdir(image_folder)
        self.image_folder = os.getcwd()
        self.images_filenames = [f for f in os.listdir(self.image_folder)]
        self.unprocessed_images = []
        self.max_img_height = 0
        self.max_img_width = 0
        self.val2idx = None
        self.idx2val = None
        self.classes_filenames_dict = {}

        if not (classes_csv or classes_lst):
            sys.exit('Provided neither csv filename or Python list for image classes')

        for image_filename in self.images_filenames:
            img = cv2.imread(image_filename)
            if img is None:
                self.unprocessed_images.append(image_filename)
            else:
                image_shape = img.shape
                if image_shape[0] > self.max_img_height:
                    self.max_img_height = image_shape[0]
                if image_shape[1] > self.max_img_width:
                    self.max_img_width = image_shape[1]

        self.processable_images = [filename for filename in self.images_filenames if
                                   filename not in self.unprocessed_images]

        if load_encoding_fname:
            encoding_dicts = joblib.load(load_encoding_fname)
            self.val2idx = encoding_dicts['val2idx']
            self.val2idx = encoding_dicts['idx2val']
        else:
            if classes_csv:
                with open(classes_csv) as csvfile:
                    data = csv.reader(csvfile)
                    # skip header
                    next(data, None)
                    self.classes = [row[0] for row in data]
            elif classes_lst:
                self.classes = classes_lst
            self.val2idx, self.idx2val = create_encode_dicts(self.classes)
            if save_encoding_fname:
                joblib.dump({'val2idx': self.val2idx, 'idx2val': self.idx2val}, save_encoding_fname)
            for class_name, index in self.val2idx.items():
                for image_filename in self.processable_images:
                    if class_name in image_filename:
                        if index not in self.classes_filenames_dict:
                            self.classes_filenames_dict[index] = []
                        self.classes_filenames_dict[index].append(image_filename)

    def load_images(self, pad=True, stretch=True, normalise='dataset'):
        if not self.classes_filenames_dict:
            sys.exit('Classes have not been loaded into Image Processor')
        else:
            loaded_images = []
            output_vectors = []
            for class_index, class_files in self.classes_filenames_dict.items():
                print('loading images for:', self.idx2val[class_index])
                for class_file in class_files:
                    output_vec = [0] * len(self.idx2val)
                    output_vec[class_index] = 1
                    file_img = cv2.imread(class_file)
                    final_image = resize_image(file_img, pad=pad, stretch=stretch)
                    if normalise == 'image':
                        # For per image per channel normalisation
                        normalised_image = normalise_image(final_image)
                        loaded_images.append(np.array(normalised_image))
                    loaded_images.append(np.array(final_image))
                    output_vectors.append(np.array(output_vec))
            if normalise == 'dataset':
                loaded_images = normalise_dataset(np.array(loaded_images))
            else:
                loaded_images = np.array(loaded_images)
            output_vectors = np.array(output_vectors)
            return loaded_images, output_vectors


def normalise_image(img, channels=3):
    output_image = None
    for ch_num in range(channels):
        ch_image = img[:, :, ch_num]
        normalised_image = (ch_image - ch_image.mean()) / ch_image.std()
        if ch_num == 0:
            output_image = normalised_image
        else:
            output_image = np.dstack((output_image, normalised_image))
    return output_image


def normalise_dataset(dataset, channels=3):
    ch_dataset = None
    output_dataset = None
    for ch_num in range(channels):
        if channels > 1:
            ch_dataset = np.expand_dims(dataset[:, :, :, ch_num], axis=3)
        elif channels == 1:
            ch_dataset = dataset
        normalised_dataset = (ch_dataset - ch_dataset.mean()) / ch_dataset.std()
        if ch_num == 0:
            output_dataset = normalised_dataset
        else:
            output_dataset = np.concatenate((output_dataset, normalised_dataset), axis=3)
    return output_dataset


def resize_image(img, pad=True, stretch=True, resized_width=200, resized_height=200):
    resized_aspect_ratio = float(resized_width)/resized_height
    (height, width, channel) = img.shape
    if stretch or float(width)/height == resized_aspect_ratio:
        output_image = cv2.resize(img, (resized_width, resized_height))
    else:
        if height > width:
            ratio = height/resized_height
            output_image = cv2.resize(img, (int(width / ratio), resized_height))
            if pad:
                padding_pixel = [0, 0, 0]
                padding_size = resized_width - int(width/ratio)
                left, right = __padding_size_split__(padding_size)
                output_image = cv2.copyMakeBorder(output_image, 0, 0, left, right,
                                                  cv2.BORDER_CONSTANT, value=padding_pixel)
        else:
            ratio = width/resized_width
            output_image = cv2.resize(img, (resized_width, int(height / ratio)))
            if pad:
                padding_pixel = [0, 0, 0]
                padding_size = resized_height - int(height / ratio)
                top, bottom = __padding_size_split__(padding_size)
                output_image = cv2.copyMakeBorder(output_image, top, bottom, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=padding_pixel)
    return output_image


def __padding_size_split__(padding_size):
    if padding_size % 2:
        return int((padding_size/2) - 0.5), int((padding_size/2) + 0.5)
    else:
        return int(padding_size/2), int(padding_size/2)


def create_encode_dicts(iterable):
    val2idx = None
    if isinstance(iterable, list):
        val2idx = {value: index for index, value in enumerate(iterable)}
    elif isinstance(iterable, dict):
        val2idx = {value: index for index, value in enumerate(iterable.keys())}
    idx2val = {index: value for value, index in val2idx.items()}
    return val2idx, idx2val


def encode_val(value, val2idx_dict):
    return val2idx_dict[value]


def decode_idx(index, idx2val_dict):
    return idx2val_dict[index]


def encode_dict_keys(dict2encode, val2idx_dict):
    for old_key in dict2encode.keys():
        dict2encode[val2idx_dict[old_key]] = dict2encode.pop(old_key)
    return dict2encode


if __name__ == '__main__':
    processor = ImageProcessor('images', '../image_scrapper/doggotime_breeds - original.csv')
    processor.load_images(pad=True)
