# Other Topics

## Semi-Supervised Learning

* The contest provides 50000 no label images for training.
They are 5 times more than labeled training images.
If we can use them effectively, there is a possibility of improving test accuracy.
In real situation, labeling by hand is expansive,
so there are many cases that has small labeled dataset and large no-labeled dataset for training.
The field is called as "Semi-Supervised Learning".

* I did the following experiment.
    * Train resnet using labeled data with Keras.
    * Using trained model, predict labels for unlabeled training data. I call this label as "Pseudo Label".
    * Re-train resnet using all 10000 labeled images and 5000 pseudo labeled images. 
    It is said that the good ratio of labeled and pseudo-labeled data is 2:1.
    * This method is called as Pseudo Labeling.
    * As a result, validation accuracy improved, but test accuracy did not improve.
        * Even in 10000 pseudo-labeled data, I got the same result.
        * It is natural that accuracy can not be improved even if what is already predictable is added as training data.
    * Based on the experimental results, we did not use unlabelled images in this contest.
    
## Slack

* Deep Learning takes time to train, so I often wait. I made a system that notifies Slack when learning is completed.
* The following is a Slack log when Hyper parameter is randomly searched. You can check this log on the go.

![](README_images/slack.png)

## Random Search of Hyper Parameters

* I tried the following hyper parameter adjustment.
    * epoch for top layer 
    * epoch for fine-tune
    * learning rate
        * Multiplying 0.1 in the middle of training improved test accuracy.
    * momentum
        * Default value is 0.8, but smaller value is better. For example, 0.75.
    * batch size
        * It is said that more value is better, but there is GPU memory limitation.
    * freeze index
        * It is number of layers that freeze in fine-tune. Freezed layer's weights are not updated in training.
        * When CNN layer number is N, It seemed that one-third of N is good for freeze index.
* It seemed that freeze index has a big influence on test accuracy, 
so from the middle, I just random searched it. 
* I tried 5-fold cross validation and fit hyper parameter on validation error, 
then I integrated their predictions on test data. The method improved test accuracy.
* It is good point of random search that you can feel free to run and stop like a lottery.

## Keras

* At first, I got started with Keras because it has good documents. I recommend Keras for beginner.
* Keras published imagenet trained models, such as vgg16, resnet, xception.
* Hypothesis
    * The Ensemble of various CNN models improve test accuracy because they have diversity.
* Verified Result
    * Using vgg16, vgg19, inception-v3, inception-v4, resnet and xception,
    I weighted them equally and I summed up their predictions.
    * Ensemble is better than each single model, and I got 0.79 accuracy on test data.
    * I recommend resnet as single model because image size is 224x224 that is smaller than xception and 
    training is fast and test accuracy is high.
* I also took in inception v4 from https://github.com/kentsommer/keras-inceptionV4


## mxnet

* I decided to use it for resnext trained model
* After training, I merged its prediction and Keras models predictions.
Exmeriments on weight averaged prediction revealed that resnet only got best test accuracy.
* I got test accuracy score 0.82 during the contest, and got 0.81639 after contest as a final score.
* I won the second place in accuracy.

## Trained Models

* Fine tuning of published models trained by imagenet is a effective method for image classification.
* I also used imagenet-11k trained model hoping for improvement because it is trained other dataset and has diversity.
But it has no positive effect.
* I am grateful for overseas trained models and 
I also thought that Japanese researcher should also release learned models.
* For example, model trained with food images, or with hiragana, or chinese characters.

## My approach to the contest

* Read Keras and mxnet documentation
* Read effective approaches for image classification contest of Kaggle
* Read papers of CNN image classification (e.g. VGG16) and understand image pre-processing
* Train, submit and leave effective methods

## Data Observation

* For each label, I placed the image in a grid and observed the nature of the data.
    * Some foods are not in the center of the image
    * Cooking is free and has big diversity. Some food has faces.
    * Some images have frames or letters attached by image processing by the user.
    
![](README_images/class_2_bread_sweets.png)

## Dish (Vessel) Recognition

* Foods are often on dish, so recognizing dish and cropping the region as input image may improve accuracy.
* I assumed that dish is a regular circle
* While changing the Y axis scale of the image, circle detection was performed using Hough transform
* As a result, it could not detect unless it was a perfect circle.
* I got 1 dish image from about 10 images. I added that dish cropped image as training data,
but accuracy did not imporove. There are two possible reasons.
    * False detection of dish recognition
    * Similar things are already met by Data Augmentation
* If I make dish detectors with CNN, I will be able to detect dishes other than regular circles

### About Main Topics

* [README.en.md](./README.en.md)