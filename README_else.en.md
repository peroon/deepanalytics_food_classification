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
    * fine tune前に加えたtop layerのみの学習epoch
    * fine tune時のepoch
    * learning rate: 途中から0.1倍すると精度が上がった
    * momentum: デフォルト0.8だが0.75など小さいほうがよさそうだった
    * batch size: 大きい方がよさそうだったがGPUのメモリに制限される
    * freeze index: fine tune時にfreezeさせて重みを更新しない層数。全総数をNとすると、1/3層あたりがよさそうだった
* 上記のうちfreeze indexの影響が大きかったので、途中からはそれに絞って探索した
* 5 fold cross validationを行い、各5モデルのvalidation errorが低くなるようにしてから予測を統合したら精度が上がった
* 外出するときにくじ引き感覚で実行し、好きなときに止められるのが、ランダムサーチの良い点である

## Keras

* 最初はKerasで実験を行った。ドキュメントが丁寧なので、最初はKerasから始めるのがオススメ
* vgg16からresnet, xceptionなどのimagenet学習済みモデルが公開されている
* 仮説
    * ネットワーク構造の違うCNNを使うことで多様性ができ、それらをアンサンブルすることで精度が向上するのではないか
* 検証結果
    * vgg16, vgg19, inception-v3, inception-v4, resnet, xceptionを用い、各モデルの予測の重み付けは同一とした
    * 単体モデルよりもアンサンブルした方が精度が向上し、0.79まで出せた
* resnetが画像サイズが224x224で扱いやすく、訓練も速くて精度も高かったので、1モデル選ぶならresnetが良い
* 公式以外から、inception v4も取り込んだ https://github.com/kentsommer/keras-inceptionV4


## mxnet

* resnextの学習済みモデルために導入
* 学習後、Kerasの各モデルの予測と合成し、重み付け和も試したが、最終的にはresnextのみを用い、コンテスト中の精度は0.82, 
コンテスト後は最終スコア0.81639で2位だった

## 学習済みモデル

* imagenetで学習したモデルのfine tuningが、画像認識に有効な手法であり、公開されている重みを利用した
* imagenet-11kで学習したモデルを足したが、精度は向上しなかった。imagenetとは違うデータセットで学習することで多様性が出ると思ったがそうではなかった
* 海外の公開済みモデルにお世話になるばかりではなく、日本も学習済みモデルの公開をすべきと思った
* 大量の料理画像を学習した料理認識用のモデルや、ひらがなや漢字を分類するモデルの重みを公開してはどうか

## コンテストへの取り組み方

* Keras, mxnetの公式サイトを読む
* KaggleなどのImage classification系の工夫を読む
* Image classification系論文(VGG等)の画像前処理を読む
* 学習を回してSubmitを繰り返し、うまく行った手法を積み重ねる

## データの観察

* ラベルごとに画像をグリッド状に配置して1枚にし、データの傾向を観察した
    * 料理が画像の中央に写っていないものがある
    * 料理は自由であり、多様性が大きく、顔がついていたりもする
    * 画像加工で枠や文字が付いているものもある
    
![](README_images/class_2_bread_sweets.png)

## 器認識

* 料理は器の上に乗っていることが多く、器を認識することができればそこだけ切り取ってより精度の高い学習データ・テストデータとすることができると考えて実験した
* 器は円であると仮定し、画像のYスケールを変化させつつ、ハフ変換の円検出を行った
* 完全な円でないと検出しづらかった
* 10枚に1枚ほど、器検出できたので、器内部だけの画像も足して学習させたが、精度は向上しなかった。理由は2つ考えられる
    * 器検出の誤検出のため
    * Data Augmentationですでに満たされている
* 器検出器をCNNで実装できれば、円以外の器も検出できるだろう

### 再現コードとモデリング詳細について

* [README.md](./README.jp.md)