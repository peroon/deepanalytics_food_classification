# (2)料理分類部門の再現コード

* Competition URL: https://deepanalytics.jp/compe/31

## 問題設定

* 25 class image classification
* 10000 labeled train images
* 10000 test images

## ソースコード

* 本プロジェクトに付属する

### 前処理方法

### 学習方法

### 予測方法

## 学習済みモデル

* 1モデル165MB
* 5モデルあるので、1GBほどになり、Githubでの管理は避けたい
* xxxに配置した

## 実行環境

* OS: Windows 10
* Python: 3.5
* 
* フレームワーク: mxnet
* メモリ: 64GB
* GPU: GTX 1080
* IDE: PyCharm

## モデリング詳細

* 画像の前処理
    * 1枚の画像を以下の方法で4枚に増幅させる（画像の横幅をW, 縦幅をHとする)
        * Cropping
            * 横長画像の場合、正方形H x Hの画像を左端、中央、右端から切り取る
            * 縦長画像の場合、正方形W x Wの画像を上端、中央、下端から切り取る
        * Scaling
            * 横長画像の場合、H x Hに強制的にリサイズする
            * 縦長画像の場合、W x Wに強制的にリサイズする
    * 得られた画像は、CNNへの入力サイズである224x224にリサイズし、PNGで保存する
    * Croppingの理由は、料理が中央以外に写っていても眼瞼に学習・予測するため
    * Scalingの理由は、Croppingでは一部削られてしまう情報を保持したまま正方形画像を作るため
    * 訓練画像10000枚を正方形にスケールしたものより、本手法で40000枚にしたほうが精度が上がることを確認した
    
* 画像のrec化
    * mxnetの学習データ形式として、画像をパックしてrecとする
    * 訓練画像
        * 40000枚を5分割し、4/5のみ(32000枚)をrec化したものを5つ作る
    * テスト画像
        * 正方形画像の作り方ごとに分けた10000枚ごとにrec化し、4つ作る
    
* 学習
    * 訓練用のrec 5つをresnext 101のfine-tuningで学習し、5つの学習済みweightを作成する
    
* 予測
    * 5つのモデルそれぞれがテスト画像の4 recに対して確率でクラスを予測する
    * 25クラス分類なので10000 x 25の行列が20できる
    * 20個の行列を加算し、行ごとに最大値を最終予測クラスとする



* resnext 101 http://data.dmlc.ml/mxnet/models/imagenet/resnext/ のfine-tuning
* 

## モデリングに対する説明

* xxx

## モデル再現のための手順

* 必要なPythonパッケージのインストール
    * mxnet (機械学習)
        * http://mxnet.io/get_started/windows_setup.html を参考にimport mxnetできるように設定します
        * ビルド済みのものが提供されています https://github.com/dmlc/mxnet/releases
        * 1つ注意すべきなのは、setupenv.cmdでの環境変数設定は、1024文字の長さ制限により、環境変数を破壊しうるということ。環境変数PATHの文字列はバックアップしておくべきです
    * cv2 (画像処理)
    * numpy (行列計算)
    * tqdm (進捗表示)

### 前準備としての画像の加工

```
提供されているデータセットのzipを展開し、ラベルと一緒に1つのディレクトリに置く。例えば以下のように。

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
    * prepare.py内のDEBUG = Trueとすることで小さいデータでの動作確認もできる
    * 最終的にはDEBUG = Falseで実行する。最新のPCで2hくらいかかります
    
```
完了後


```
    
* 学習と予測
    * 上記で作ったrecファイルを画像データとして、学習・予測する
    * dmlc_mxnet\example\image-classification\fine-tune.py 参照

## 教師なし学習について

* 本コンテストでは、ラベルなし画像が50000枚提供されている。ラベルあり画像の5倍の量があり、うまく利用できれば精度を向上できる可能性がある。
現実問題としてもラベルの手付けはコストがかかるので、
少数のラベル付きデータと、多数のラベルなしデータから学習させたいシチュエーションは多く、半教師あり学習(Semi-supervised Learning)
という研究分野である。

* 以下の実験を行った
    * Kerasのresnetをラベル付き訓練データで学習する
    * 学習したモデルでラベルなし画像に対してpredictし、ラベルを付ける。これを擬似ラベルと呼ぶとする
    * ラベル付き画像10000枚、疑似ラベル付き画像5000枚でresnetを再度学習する (ラベル : 疑似ラベルの割合は2:1が良いと言われている)
    * これはPseudo Labelingと言われている
    * 結果、validation accuracyは上がったが、test accuracyは上がらなかった
        * 疑似ラベル付き画像を10000枚にしても同様
        * すでに予測できるものを学習画像として加えても精度が上がらないのは、それはそうだよねと思える
    * 実験結果から、本コンテストではラベルなし画像を使用しなかった