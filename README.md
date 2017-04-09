# (2)料理分類部門の再現コードとモデリング詳細

* Competition URL: https://deepanalytics.jp/compe/31

## 問題設定

* 25 class image classification
* 10000 labeled train images
* 10000 test images

## ソースコード

* 本プロジェクトに付属する

## 学習済みモデル

* [Fine tuned weight files is available](https://drive.google.com/drive/folders/0BxkHqJ_0XZ-lb0s3azltRHhNVTg?usp=sharing)

## 実行環境

* OS: Windows 10
* Python: 3.5
* フレームワーク: mxnet
* メモリ: 64GB
* GPU: GTX 1080
* IDE: PyCharm

## 提出したレポート

* [Speaker Deck](https://speakerdeck.com/peroon/food-image-classification)
* 工夫点など、10ページでまとめてあります

## モデリング詳細

* 画像の前処理
    * 前提
        * CNNへの入力として正方形画像にしたい
    * 仮説
        * 中央に写っていない料理があるので、上下左右から切り取った画像を学習画像にすると位置に頑健となり精度が上がるのではないか
        * 切り取らず、正方形に縮小させた画像も情報量がほぼ保持されるので、良い正方形画像の作り方なのではないか
    * 手法
        * 1枚の画像を以下の方法で4枚に増幅させる（画像の横幅をW, 縦幅をHとする)
            * Cropping
                * 横長画像の場合、正方形H x Hの画像を左端、中央、右端から切り取る
                * 縦長画像の場合、正方形W x Wの画像を上端、中央、下端から切り取る
            * Scaling
                * 横長画像の場合、H x Hに強制的にリサイズする
                * 縦長画像の場合、W x Wに強制的にリサイズする
        * 得られた画像を、CNN(resnext)への入力サイズである224x224にリサイズし、PNGで保存する
    * 検証結果
        * 訓練画像10000枚を正方形にスケールしたものより、本手法で40000枚にした方が精度が向上した
    
* 画像セットを分割してrec化する
    * 前提
        * mxnetの学習データ形式として、画像セットをパックしてrecファイルにまとめる
    * 仮説
        * 画像をすべて用いて学習した1モデルより、クロスバリデーションの作法で訓練データを分割し、
        それを学習した複数モデルの予測を合わせると精度が向上するのではないか
    * 分割手法
        * 訓練画像
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

* 必要なPythonパッケージのインストール
    * mxnet (機械学習)
        * http://mxnet.io/get_started/windows_setup.html を参考にimport mxnetできるように設定します
        * ビルド済みのものが提供されています https://github.com/dmlc/mxnet/releases
        * ※注意 上記で提供されているsetupenv.cmdでの環境変数設定は、1024文字の長さ制限により、環境変数を破壊しうるので、
        環境変数PATHの文字列はバックアップしておくべき
    * cv2 (画像処理)
    * numpy (行列計算)
    * tqdm (進捗表示)

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
完了後、下記のようにrecファイルが出力される

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
    
* 学習
    * 上記で作ったrecファイルを画像データとして、学習・予測する
    * dmlc_mxnet\example\image-classification\fine-tune.py 参照
    * 計算時間は、GTX1080で20epoch 5hかかり、それを5モデル分行うので25hかかる
    * 1モデル作成するごとにcudnn memory errorで落ちるので、訓練は5回実行してください。
    その際、下記コードの0の部分を1, 2, 3, 4と書き換えてください。
        * 本件の[Issue](https://github.com/peroon/deepanalytics_food_classification/issues/1)

```
#fine-tune.py 

if mode == 'train':
    for fold_i in range(0, 5):
```

* 予測
    * recファイルに対して予測する
    * dmlc_mxnet\example\image-classification\fine-tune.py 参照
    * GTX1080で、10000枚の確率予測を20セット(4 crop x 5 model)行い、それぞれの予測結果をnpyで保存する
    * 最後に確率予測の平均を取り、最大のラベルに予測する

### 再現コードとモデリング詳細以外について

* [README_else.md](./README_else.md)