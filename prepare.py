# -*- coding: utf-8 -*-

import glob
import cv2
import os
import shutil

from constant import IMAGES_ROOT


def make_directory():
    os.mkdir(IMAGES_ROOT + 'clf_train_all')
    os.mkdir(IMAGES_ROOT + 'clf_train_224x224_0')
    os.mkdir(IMAGES_ROOT + 'clf_train_224x224_1')
    os.mkdir(IMAGES_ROOT + 'clf_train_224x224_2')
    os.mkdir(IMAGES_ROOT + 'clf_train_224x224_3')

    os.mkdir(IMAGES_ROOT + 'clf_test_all')
    os.mkdir(IMAGES_ROOT + 'clf_test_224x224_0')
    os.mkdir(IMAGES_ROOT + 'clf_test_224x224_1')
    os.mkdir(IMAGES_ROOT + 'clf_test_224x224_2')
    os.mkdir(IMAGES_ROOT + 'clf_test_224x224_3')


def __new_filename(filename='train_1.jpg'):
    image_id = int(filename.split('.')[0].split('_')[-1])
    return '%08d' % image_id + '.jpg'


def copy_and_rename():
    for name in ['train', 'test']:
        image_path_list = glob.glob(IMAGES_ROOT + 'clf_{}_*/*.jpg'.format(name))
        output_directory = IMAGES_ROOT + 'clf_{}_all/'.format(name)
        for i, image_path in enumerate(image_path_list):
            if i % 1000 == 0:
                print('copy', name, i) # progress
            filename = os.path.basename(image_path)
            new_filename = __new_filename(filename)
            copy_dst_path = output_directory + new_filename
            shutil.copy(image_path, copy_dst_path)


def crop():
    pass


def resize():
    pass


def make_lst():
    pass


def make_rec():
    pass


if __name__ == '__main__':
    #make_directory()
    #copy_and_rename()
    #crop()
    #resize()
    #make_lst()
    #make_rec()