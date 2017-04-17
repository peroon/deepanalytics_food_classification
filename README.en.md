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
                * I regard left-end-cropped and top-end-cropped as same.
                * right-end-cropped and bottom-end-cropped, too.
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
            * Each 10000 training images of 4 sets are splited by 5 (8000 images and 2000 images)
            * 8000 images x 4 set = 32000 images. They are packed to rec file.
            * According to Cross Validation like split, I make 5 set of 32000 images rec.
        * Validation Images
            * I make 5 set of 8000 (2000 images x 4 type) images rec.
        * Test Images
            * I don't need split test images. Because it is not used training and ensembling.
            * 10000 test images of 4 types are each packed as rec file.
    * Verification
        * I confirmed our ensemble method (B) is superior to normal training (A) on test precision.
    
* Training
    * Hypothesis
        * I made a progress by ensembling many CNN models such as vgg16, vgg19, resnet, inception-v3, xception. 
        I assumed that resnext is also good for ensemble.
    * Method
        * resnext 101 http://data.dmlc.ml/mxnet/models/imagenet/resnext/
        * I fine-tuned it with 5 training set and made 5 models
    * Verification
        * After adding resnext to ensemble group, test precision was improved.
        * I checked the influence of weighted averaged prediction of each model, and found that
        resnext 5 model ensemble is best. At last, other models was not needed.
    
* Prediction
    * Method
        * 5 resnext models predict label probability on 4 set of test images (4 x 10000) individually. 
        * After each model prediction, I get 20 matrices that is 10000 x 25 probabilities, 
        because of 25 labels classification.
        * Finally, sum up 20 matrices and select max probablity on each row, and predict the label.

## Concrete sequeces of reproduction

* Python Packages
    * mxnet (machine learning)
        * To install, refer to http://mxnet.io/get_started/windows_setup.html
        * Built mxnet is provided for Windows user https://github.com/dmlc/mxnet/releases
        * [Warning] Upper site provides setupenv.cmd for auto PATH setting. 
        But Windows has 1024 characters limitation of PAHT. If the limitation occurs on executing setupenv.cmd,
        PATH is cut off, so make sure PATH backup.
    * cv2 (image processing)
    * numpy (matrix)
    * tqdm (progress bar)

### Image Pre-processing

```
Unzip provided dataset and put them in a directory with label data. 
Write the pass in constant.py.
Putting example,

C:\Users\kt\Documents\DataSet\cookpad>ls -l
total 8376
drwxr-xr-x 1 kt 197614      0 Apr  4 20:53 clf_test_images_1
drwxr-xr-x 1 kt 197614      0 Apr  4 20:54 clf_test_images_2
drwxr-xr-x 1 kt 197614      0 Apr  4 20:50 clf_train_images_labeled_1
drwxr-xr-x 1 kt 197614      0 Apr  4 20:50 clf_train_images_labeled_2
-rw-r--r-- 1 kt 197614 184920 Apr  4 15:04 clf_train_master.tsv
```
    
* To process images, execute prepare.py
    * The script does crop, resize, rename and move
    * After execution, open command prompt where lst files are, and execute make_rec.bat to make rec files.
    * You can experiment on small dataset by rewriting DEBUG = True in prepare.py
    * Making rec files takes 2 hours by 2017 latest PC.
    
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
    * This training code has a bug. 
    After training a model, it cause "cudnn memory error" so you should execute the script 5 times.
    When re-execution, rewrite 0 to 1, 2, 3, 4 of fine-tune.py
        * [Issue](https://github.com/peroon/deepanalytics_food_classification/issues/1)

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

### Other Topics

* [README_else.en.md](./README_else.en.md)