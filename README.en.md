# Reproduction Code of Food Classification & Modeling Details

* Competition URL: https://deepanalytics.jp/compe/31

## Problem Settings

* 25 class image classification
* 10000 labeled train images
* 10000 test images

## Soure Code

* exists in this project.

## Trained Model

* [Fine tuned weight files is available](https://drive.google.com/drive/folders/0BxkHqJ_0XZ-lb0s3azltRHhNVTg?usp=sharing)

## Execution Environment

* OS: Windows 10
* Python: 3.5
* Framework: mxnet
* Memory: 64GB
* GPU: GTX 1080
* IDE: PyCharm

## Submitted Report

* [Speaker Deck](https://speakerdeck.com/peroon/food-image-classification)
* 10 pages report about creative points (Japaneese)

## Modeling Details

* Pre-process of Images
    * Premise
        * I need square image as input for CNN
    * Hypothesis
        * I noticed that food is not always center positioned in a image, so cropping of image may work well. 
        It will give the robustness of food position.
        * Scaling to square without retaining the original aspect ratio also may work well. 
        Compared to cropping, it discard less information. 
    * Method 
        * This method creates four images from a original image
        * W: width of image
        * H: height of image
            * Cropping
                * If W >= H, crop H x H square iamges from left end, center and right end.
                * If W <H, crop W x W square iamges from top end, center and bottom end.
            * Scaling
                * If W >= H, resize to H x H size image
                * If W < H, resize to W x W size image
        * These images are resized t0 224x224 and save as PNG. 
        They are input for CNN (resnext).
    * Verification
        * A: Train CNN by 10000 square-scaled images
        * B(ours): Train CNN by 40000 image got by our method
        * B > A about test data accuracy
        
* Split images and make rec files
    * Premise
        * As input data strudcture for mxnet, pack images as a rec file.
    * Hypothesis
        * A: 1 model trained by all training images
        * B(ours): Split 10000 training images into 8000 training images and 2000 validation images according to 
        Cross Validation rule. After making 5 set, 5 CNN models are trained by each 8000 training images.
        For prediction, 5 models prediction are summed.
        * I forecasted B > A
    * Split Method
        * Training Images
            * 40000枚を5分割し、4/5のみ(32000枚)をrec化したものを5セット作る
        * バリデーション画像
            * 訓練画像で余った8000枚をrec化したものを5セット作る
        * テスト画像
            * 正方形画像の作り方ごと(全4種)に分けた10000枚ごとにrec化し、4セット作る
    * 検証結果
        * 分割して複数のモデルを作ってアンサンブルした方が、40000枚で学習した1モデルよりも精度が向上した
    
* 学習
    * 仮説
        * vgg16, resnetなど様々なモデルのアンサンブルで精度を上げてきたが、resnextを足すのが良いのではないか
    * 手法
        * resnext 101 http://data.dmlc.ml/mxnet/models/imagenet/resnext/ のfine-tuning
        * 訓練用データ5セットをresnext 101のfine-tuningで学習し、5つの学習済みモデルを作成する
    * 検証結果
        * resnextをアンサンブルに加えることで精度が上がった
        * vgg16, resnet, resnextなど各モデルの重み付けを変えてテストの精度を観察したところ、
        最終的にはresnext単体の5モデルでアンサンブルすると1番精度が高かった
    
* 予測
    * 手法
        * 5つのモデルそれぞれがテスト画像の4 recに対してラベルを確率予測する
        * 25クラス分類なので10000 x 25の行列が20個できる
        * 20個の行列を加算し、行ごとに最大確率のラベルを予測ラベルとする


## モデル再現のための手順

* Needed Python Packages
    * mxnet (machine learning)
        * http://mxnet.io/get_started/windows_setup.html を参考にimport mxnetできるように設定します
        * Built mxnet is provided for Windows user https://github.com/dmlc/mxnet/releases
        * [Warning] 上記で提供されているsetupenv.cmdでの環境変数設定は、1024文字の長さ制限により、環境変数を破壊しうるので、
        環境変数PATHの文字列はバックアップしておくべき
    * cv2 (image processing)
    * numpy (matrix)
    * tqdm (progress bar)

### 前準備としての画像の加工

```
提供されているデータセットのzipを展開し、ラベルと一緒に1つのディレクトリに置く。例えば以下のように置き、フルパスはconstant.pyに記入する

C:\Users\kt\Documents\DataSet\cookpad>ls -l
total 8376
drwxr-xr-x 1 kt 197614      0 Apr  4 20:53 clf_test_images_1
drwxr-xr-x 1 kt 197614      0 Apr  4 20:54 clf_test_images_2
drwxr-xr-x 1 kt 197614      0 Apr  4 20:50 clf_train_images_labeled_1
drwxr-xr-x 1 kt 197614      0 Apr  4 20:50 clf_train_images_labeled_2
-rw-r--r-- 1 kt 197614 184920 Apr  4 15:04 clf_train_master.tsv
```
    
* prepare.pyを実行し、画像を加工する
    * crop, resize, rename, moveする
    * 完了後、lstファイルの位置でcmdを開き、make_rec.batを実行し、データセットrecを作る
    * prepare.py内のDEBUG = Trueとすることで小さいデータでの動作確認ができる
    * 最終的にはDEBUG = Falseで実行する。最新のPCでrec作成に2h程かかります
    
```
After completion, rec files are outptuted like this.

C:\Users\kt\Documents\DataSet\cookpad\mxnet>ls -l *.rec
-rw-r--r-- 1 kt 197614  915396960 Apr  8 19:57 test_224x224_crop_0.rec
-rw-r--r-- 1 kt 197614  933075016 Apr  8 20:01 test_224x224_crop_1.rec
-rw-r--r-- 1 kt 197614  922141600 Apr  8 20:05 test_224x224_crop_2.rec
-rw-r--r-- 1 kt 197614  923399752 Apr  8 20:09 test_224x224_crop_3.rec
-rw-r--r-- 1 kt 197614 2966390868 Apr  8 20:23 train_224x224_fold_0.rec
-rw-r--r-- 1 kt 197614 2962274168 Apr  8 20:35 train_224x224_fold_1.rec
-rw-r--r-- 1 kt 197614 2964625728 Apr  8 20:48 train_224x224_fold_2.rec
-rw-r--r-- 1 kt 197614 2963049840 Apr  8 21:00 train_224x224_fold_3.rec
-rw-r--r-- 1 kt 197614 2960390948 Apr  8 21:12 train_224x224_fold_4.rec
-rw-r--r-- 1 kt 197614  737792020 Apr  8 21:15 validation_224x224_fold_0.rec
-rw-r--r-- 1 kt 197614  741908720 Apr  8 21:18 validation_224x224_fold_1.rec
-rw-r--r-- 1 kt 197614  739557160 Apr  8 21:21 validation_224x224_fold_2.rec
-rw-r--r-- 1 kt 197614  741133048 Apr  8 21:24 validation_224x224_fold_3.rec
-rw-r--r-- 1 kt 197614  743791940 Apr  8 21:27 validation_224x224_fold_4.rec
```
    
* Training
    * Using thiese rec files as images data, Training and Prediction
    * refer to dmlc_mxnet\example\image-classification\fine-tune.py
    * To train, it spends 25 hours for 5 models and each 20 epoch. 
    One model trainig spends about 5 hours with GTX1080 
    * 1モデル作成するごとにcudnn memory errorで落ちるので、訓練は5回実行してください。
    その際、下記コードの0の部分を1, 2, 3, 4と書き換えてください。
        * 本件の[Issue](https://github.com/peroon/deepanalytics_food_classification/issues/1)

```
#fine-tune.py 

if mode == 'train':
    for fold_i in range(0, 5):
```

* Prediction
    * refer to dmlc_mxnet\example\image-classification\fine-tune.py
    * Test data is 20 set (4 crops x 5 models) of 10000 images. Predict each set and 
    save result to each .npy files.
    * Finally, I sum up all predicted probability matrix, and according to max probability, 
    each label is decided for each image.

### 再現コードとモデリング詳細以外について

* [README_else.md](./README_else.jp.md)