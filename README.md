# YoloV3

空のディレクトリ`utils`を作成し、その中に`bbox.py`, `colos.py`, `image.py`, `multi_gpu_model.py`, `utils.py`を入れる。
空のディレクトリ`weights`を作成しておく。
YoloV3の重みファイル`weights_concrete.h5`または、pickleファイル`concrete_train.pkl`が既に存在している場合、削除する。
## アンカー作成
`gen_anchors.py`を実行することで、アンカー作成する。
入力画像フォルダと入力画像フォルダに対応するXMLフォルダのパスを設定しておく。
`python gen_anchors.py`
実行することで`concrete_train.pkl`が生成される。
## Yolov3の学習
`gen_anchors.py`の実行で生成されたアンカーに書き換える。
入力画像フォルダと入力画像フォルダに対応するXMLフォルダのパスを設定しておく。
`train.py`または、`Yolo_v3-hflip.ipynb`を実行することで、YoloV3を学習させる。
`python train.py`
実行することでYolov3の重み`weights_concrete.h5`が生成される。
