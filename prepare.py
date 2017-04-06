# -*- coding: utf-8 -*-

import glob
import cv2
import os
import shutil
from collections import deque

from tqdm import tqdm

from constant import IMAGES_ROOT
from constant import IMAGE_SIZE
from square_cropper import SquareCropper

DEBUG = True
mxnet_dir = IMAGES_ROOT + 'mxnet/'


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
    """元々のファイル名を 8桁 zero paddingしたファイル名に変更"""

    image_id = int(filename.split('.')[0].split('_')[-1])
    return '%08d' % image_id + '.jpg'


def copy_and_rename():
    """元々の画像ファイルをリネームしながら1フォルダにまとめる"""

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
    """画像をcrop, resizeする"""

    cropper = SquareCropper()

    for name in ['train', 'test']:
        image_path_list = glob.glob(IMAGES_ROOT + 'clf_{}_all/*.jpg'.format(name))
        for image_path in tqdm(image_path_list):
            image = cv2.imread(image_path)
            cropped_image_list = cropper.crop_all(image, IMAGE_SIZE)
            for crop_index, cropped_image in enumerate(cropped_image_list):
                save_dir = IMAGES_ROOT + 'clf_{}_224x224_{}/'.format(name, crop_index)
                filename = os.path.basename(image_path).replace('.jpg', '.png')
                cv2.imwrite(save_dir + filename, cropped_image)


def split_list(list_input, fold_num=5, fold_index=0):
    """split data for train and validation"""

    num_all = len(list_input)
    num_val = int(num_all / fold_num)
    num_tra = num_all - num_val

    deq = deque(list_input)
    deq.rotate(fold_index * num_val)
    return list(deq)[:num_tra], list(deq)[num_tra:]


def get_labels():
    labels = []
    with open(IMAGES_ROOT + 'clf_train_master.tsv', 'r') as f:
        f.readline()
        for s in f:
            label = int(s.split()[-1])
            labels.append(label)
    return labels


def write_lst(save_path, image_path_list, labels):
    f = open(save_path, 'w')
    for i in range(len(image_path_list)):
        s = '{}\t{}\t{}\n'.format(i, labels[i], image_path_list[i])
        f.write(s)
    f.close()


def make_train_lst():
    if not os.path.exists(IMAGES_ROOT + 'mxnet'):
        os.mkdir(IMAGES_ROOT + 'mxnet')

    for fold_i in range(5):
        # foldごとにデータリストを作る
        image_path_list_train_all = []
        image_path_list_validation_all = []
        labels_train_all = []
        labels_validation_all = []

        for crop_i in range(4):
            glob_path = IMAGES_ROOT + 'clf_train_224x224_{}/*'.format(crop_i)
            image_path_list = glob.glob(glob_path)

            if DEBUG:
                image_path_list = image_path_list[:100]

            # train/valにsplitする
            labels = get_labels()
            image_path_list_train, image_path_list_validation = split_list(image_path_list, fold_index=fold_i)
            labels_train, labels_validation = split_list(labels, fold_index=fold_i)

            # allに加える
            image_path_list_train_all += image_path_list_train
            image_path_list_validation_all += image_path_list_validation
            labels_train_all += labels_train
            labels_validation_all += labels_validation

        # ファイルに書き込む
        train_lst_path = mxnet_dir + 'train_224x224_fold_{}.lst'.format(fold_i)
        write_lst(train_lst_path, image_path_list_train_all, labels_train_all)

        validation_lst_path = mxnet_dir + 'validation_224x224_fold_{}.lst'.format(fold_i)
        write_lst(validation_lst_path, image_path_list_validation_all, labels_validation_all)


def make_test_lst():

    # 各cropごとに1ファイルにする
    for crop_i in range(0, 4):
        glob_path = IMAGES_ROOT + 'clf_test_224x224_{}/*'.format(crop_i)
        image_path_list = glob.glob(glob_path)

        if DEBUG:
            image_path_list = image_path_list[:100]

        labels = [0] * len(image_path_list)

        # ファイルに書き込む
        lst_path = mxnet_dir + 'test_224x224_crop_{}.lst'.format(crop_i)
        write_lst(lst_path, image_path_list, labels)


def make_rec():
    im2rec_fullpath = os.path.abspath('./dmlc_mxnet/tools/im2rec.py')
    lst_path_list = glob.glob(mxnet_dir + '*.lst')

    # make bat
    with open(mxnet_dir + 'make_rec.bat', 'w') as f:
        for lst_path in lst_path_list:
            filename_without_ext = os.path.basename(lst_path).split('.')[0]
            print(filename_without_ext)
            s = 'python {} --encoding .png {} .'.format(im2rec_fullpath, filename_without_ext)
            f.write(s + '\n')

    print('Execute .bat to make rec file.')
    print(mxnet_dir)




if __name__ == '__main__':
    # make_directory()
    # copy_and_rename()
    # crop()
    # make_train_lst()
    # make_test_lst()
    # make_rec()
    pass