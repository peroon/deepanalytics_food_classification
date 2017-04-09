# -*- coding: utf-8 -*-

import os
import urllib.request
import time
import glob

import mxnet as mx
import numpy as np

from constant import IMAGES_ROOT


def make_directory():
    os.makedirs('./temp/model_weight/resnext-101', exist_ok=True)
    os.makedirs('./temp/predict/resnext-101', exist_ok=True)
    os.makedirs('./temp/submit', exist_ok=True)


def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)


def get_model(prefix, epoch):
    download(prefix + '-symbol.json')
    download(prefix + '-%04d.params' % (epoch,))


def get_iterators(batch_size, data_shape=(3, 224, 224), fold_index=0):
    train_rec = IMAGES_ROOT + "mxnet/train_224x224_fold_{}.rec".format(fold_index)
    valid_rec = IMAGES_ROOT + "mxnet/validation_224x224_fold_{}.rec".format(fold_index)

    train = mx.io.ImageRecordIter(
        path_imgrec=train_rec,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True)

    val = mx.io.ImageRecordIter(
        path_imgrec=valid_rec,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False)
    return (train, val)


def fine_tune(mode):
    BATCH_PER_GPU = 20
    CLASS_NUM = 25
    EPOCH_NUM = 20
    MODEL_NAME = 'resnext-101'
    ENABLE_VALIDATION = True
    GPU_NUM = 1

    print('学習済みモデルをDL', MODEL_NAME)
    if MODEL_NAME is 'resnext-101':
        get_model('http://data.mxnet.io/models/imagenet/resnext/101-layers/resnext-101', 0)
        sym, arg_params, aux_params = mx.model.load_checkpoint('resnext-101', 0)

    def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
        """
        symbol: the pre-trained network symbol
        arg_params: the argument parameters of the pre-trained model
        num_classes: the number of classes for the fine-tune datasets
        layer_name: the layer name before the last fully-connected layer
        """
        all_layers = sym.get_internals()
        net = all_layers[layer_name + '_output']
        net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
        net = mx.symbol.Dropout(data=net, p=0.50000)
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
        new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
        return (net, new_args)

    # 学習中のログ
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, CLASS_NUM)

    def fit(symbol, arg_params, aux_params, train, val, batch_size, GPU_NUM, fold_index):
        devs = [mx.gpu(i) for i in range(GPU_NUM)]
        mod = mx.mod.Module(symbol=new_sym, context=devs)
        mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        mod.set_params(new_args, aux_params, allow_missing=True)

        start_time = time.time()
        mod.fit(train,
                eval_data=val,
                num_epoch=EPOCH_NUM,
                batch_end_callback=[
                    mx.callback.Speedometer(batch_size, 10),
                ],
                kvstore='device',
                optimizer='sgd',
                optimizer_params={
                    'learning_rate': 0.001,
                },
                eval_metric='acc')
        end_time = time.time()
        passed_minutes = int((end_time - start_time) / 60)
        print('学習完了', 'min', passed_minutes)
        mod.save_params('temp/model_weight/{}/model_params_fold_i_{}'.format(MODEL_NAME, fold_index))

        if ENABLE_VALIDATION:
            metric = mx.metric.Accuracy()
            return mod.score(val, metric)
        else:
            return None

    if mode == 'train':
        for fold_i in range(0, 5):
            print('学習します', 'fold i', fold_i)
            batch_size = BATCH_PER_GPU * GPU_NUM
            (train, val) = get_iterators(batch_size, fold_index=fold_i)
            mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, GPU_NUM, fold_index=fold_i)
            if mod_score:
                result = list(mod_score)
                print('result', result)
                acc = result[0][1]
                print('学習モデルの評価結果', acc)

    if mode == 'predict':
        for fold_i in range(5):
            for crop_i in range(4):
                print('予測します', 'fold i', fold_i, 'crop i', crop_i)

                batch_size = BATCH_PER_GPU * GPU_NUM
                test_rec_path = IMAGES_ROOT + "mxnet/test_224x224_crop_{}.rec".format(crop_i)
                test = mx.io.ImageRecordIter(
                    path_imgrec=test_rec_path,
                    data_name='data',
                    label_name='softmax_label',
                    batch_size=batch_size,
                    data_shape=(3, 224, 224),
                    rand_crop=False,
                    rand_mirror=False)

                devs = [mx.gpu(i) for i in range(GPU_NUM)]
                mod = mx.mod.Module(symbol=new_sym, context=devs)
                mod.bind(data_shapes=test.provide_data, label_shapes=test.provide_label)
                mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
                mod.set_params(new_args, aux_params, allow_missing=True)

                # 学習済みパラメータをロード
                mod.load_params('temp/model_weight/{}/model_params_fold_i_{}'.format(MODEL_NAME, fold_i))
                output_list = mod.predict(eval_data=test, num_batch=None)

                # 予測して確率を保存
                probabilities = output_list.asnumpy()
                save_path = 'temp/predict/{}/fold_{}_crop_{}_predict.npy'.format(MODEL_NAME, fold_i, crop_i)
                np.save(save_path, probabilities)


def average_predict():
    """average predictions and make a final prediction"""

    # 強識別器
    predict_path_list = glob.glob('temp/predict/resnext-101/*.npy'.format())

    predict = np.load(predict_path_list[0])
    predict_sum = np.zeros(predict.shape, dtype='float32')

    for predict_path in predict_path_list:
        predict = np.load(predict_path)
        predict_sum += predict

    with open('./temp/submit/predict_for_submit.csv', 'w') as f:
        for i, predict in enumerate(predict_sum):
            label = np.argmax(predict)
            s = '{0},{1}'.format(i, label) + '\n'
            f.write(s)


if __name__ == '__main__':
    make_directory()
    fine_tune(mode='train')
    fine_tune(mode='predict')
    average_predict()
