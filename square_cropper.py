# -*- coding: utf-8 -*-

"""正方形にcropする"""

import cv2
from enum import Enum


class CutType(Enum):
    """切り取りタイプ"""
    upper_left = 0
    center = 1
    lower_right = 2
    compress = 3


class SquareCropper():

    def get_length(self, image):
        return min(image.shape[:2])

    # 左上から切る
    def __cut_from_upper_left(self, image):
        length = self.get_length(image)
        return image[0:length, 0:length]

    # 右下から切る
    def __cut_from_lower_right(self, image):
        h, w = image.shape[:2]
        length = self.get_length(image)
        return image[h - length:, w - length:]

    # 中央から切る
    def __cut_center(self, image):
        h, w = image.shape[:2]
        if h == w:
            return image
        # 横長
        elif w > h:
            x = int(w / 2 - h / 2)
            return image[:, x:x + h]
        # 縦長
        elif h > w:
            y = int(h / 2 - w / 2)
            return image[y:y + w, :]

    # 縦横で長い方は圧縮する
    def __cut_compress(self, image):
        length = self.get_length(image)
        return cv2.resize(image, (length, length))

    def crop(self, image, length, cut_type=CutType.center):
        if cut_type == CutType.upper_left:
            cropped = self.__cut_from_upper_left(image)
        elif cut_type == CutType.center:
            cropped = self.__cut_center(image)
        elif cut_type == CutType.lower_right:
            cropped = self.__cut_from_lower_right(image)
        elif cut_type == CutType.compress:
            cropped = self.__cut_compress(image)
        else:
            cropped = None
        return self.resize(cropped, length)

    def resize(self, image, length):
        if image is None:
            return None
        return cv2.resize(image, (length, length))

    def crop_all(self, image, length):
        image_list = []
        image_list.append(self.crop(image, length, CutType.upper_left))
        image_list.append(self.crop(image, length, CutType.center))
        image_list.append(self.crop(image, length, CutType.lower_right))
        image_list.append(self.crop(image, length, CutType.compress))
        return image_list

if __name__ == '__main__':
    # 動作テスト
    cropper = SquareCropper()
    path = "C:\\Users\\kt\\Documents\\DataSet\\cookpad\\clf_test_299x299\\00000000.png"
    image = cv2.imread(path)
    images = cropper.crop_all(image, 123)
    print(len(images))


