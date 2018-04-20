from os import listdir
from os.path import isfile, join
# import keras as k
import csv
import numpy as np
import cv2


class ImageProcessor(object):
    def __init__(self, image_folder, classes_csv=None):
        self.folder = image_folder
        self.image_filenames = [f for f in listdir(self.folder) if isfile(join(self.folder, f))]
        self.unprocessed_images = []

        self.max_img_height = 0
        self.max_img_width = 0

        for image in self.image_filenames:
            image = cv2.imread(self.folder + image)
            if image is None:
                self.unprocessed_images.append(image)
            else:
                image_shape = image.shape
                if image_shape[0] > self.max_img_height:
                    self.max_img_height = image_shape[0]
                if image_shape[1] > self.max_img_width:
                    self.max_img_width = image_shape[1]
        self.processable_images = [filename for filename in self.image_filenames if
                                   filename not in self.unprocessed_images]

        self.classes_files_dict = {}
        self.classes_images = {}

        if classes_csv:
            with open(classes_csv) as csvfile:
                data = csv.reader(csvfile)
                # skip header
                next(data, None)
                for row in data:
                    self.classes_files_dict[row[0]] = []

        if self.classes_files_dict:
            for class_name, class_files in self.classes_files_dict.items():
                for image_filename in self.processable_images:
                    if class_name in image_filename:
                        class_files.append(image_filename)

    # # filtering processable and unprocessable images
    # def __image_processability_filter__(self):
    #     for image in self.image_filenames:
    #         if cv2.imread(self.folder + image) is None:
    #             self.unprocessed_images.append(image)
    #     self.processable_images = [filename for filename in self.image_filenames if
    #                                filename not in self.unprocessed_images]
    #
    # # for padding
    # def __get_pad_dims__(self):
    #     for image in self.processable_images:
    #         image_shape = cv2.imread(self.folder + image).shape
    #         if image_shape[0] > self.max_img_height:
    #             self.max_img_height = image_shape[0]
    #         if image_shape[1] > self.max_img_width:
    #             self.max_img_width = image_shape[1]

    def load_images(self, padding=True):
        if not self.classes_files_dict:
            print('Classes have not been loaded into Image Processor')
        else:
            for class_name, class_files in self.classes_files_dict.items():
                self.classes_images[class_name] = []
                for class_file in class_files:
                    file_img = cv2.imread(self.folder + class_file)
                    # top, bottom, left, right = self.__image_pad_size__(file_img)
                    # padded_image = cv2.copyMakeBorder(file_img, top, bottom, left, right,
                    #                                   cv2.BORDER_CONSTANT, value=padding_pixel)
                    if padding:
                        final_image = self.__resize_pad_image__(file_img, pad=True)
                    else:
                        final_image = self.__resize_pad_image__(file_img, pad=False)
                    self.classes_images[class_name].append(final_image)
                np.asarray(self.classes_images[class_name])

    def __pad_size_split__(self, padding_size):
        if padding_size % 2:
            return int((padding_size/2) - 0.5), int((padding_size/2) + 0.5)
        else:
            return int(padding_size/2), int(padding_size/2)

    def __resize_pad_image__(self, image, pad=True, resized_height=800,  resized_width=800):
        (height, width, channel) = image.shape
        if height > width:
            ratio = height/resized_height
            scaled_image = cv2.resize(image, (int(width/ratio), resized_height))
            if pad:
                padding_pixel = [0, 0, 0]
                padding_size = resized_width - int(width/ratio)
                left, right = self.__pad_size_split__(padding_size)
                padded_image = cv2.copyMakeBorder(scaled_image, 0, 0, left, right,
                                                  cv2.BORDER_CONSTANT, value=padding_pixel)
                return padded_image
        else:
            ratio = width/resized_width
            scaled_image = cv2.resize(image, (resized_width, int(height/ratio)))
            if pad:
                padding_pixel = [0, 0, 0]
                padding_size = resized_height - int(height / ratio)
                top, bottom = self.__pad_size_split__(padding_size)
                padded_image = cv2.copyMakeBorder(scaled_image, top, bottom, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=padding_pixel)
                return padded_image
        return scaled_image


def create_encoding_dicts(dict_or_list):
    if isinstance(dict_or_list, dict):
        name2enc = {name: index for index, name in enumerate(list(set([name for name, value in dict_or_list.items()])))}
    elif isinstance(dict_or_list, list):
        name2enc = {name: index for index, name in enumerate(list(set(dict_or_list)))}
    enc2name = {value: key for key, value in name2enc.items()}
    return name2enc, enc2name


def encode_classes(dict):
    return None


def decode_classes(list_or_nparray):
    return None


if __name__ == '__main__':
    processor = ImageProcessor('images/', 'image_scrapper/doggotime_breeds - original.csv')
    processor.load_images(padding=True)


    # name2enc, enc2name = create_encoding_dicts([5, 5, 5, 3, 'hello', 'whatsup', 'hello'])
    # name2enc, enc2name = create_encoding_dicts({'abc': 1, 'bcd': 2, 'abc': 1, 'bee': 4})
    # print(True)