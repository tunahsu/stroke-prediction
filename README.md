# Stroke Prediction


## 簡介

* 基於 3D-CNN 的 CT 影像分類模型
* 使用 YOLOv5 偵測 CT 影像中心臟的區域
* 在訓練、測試前透過 uniformizing techniques 統一輸入影像的大小
* Guided-GradCAM 視覺化模型在 CT 影像中關注區域


## 套件

```
tensorflow
opencv-python
matplotlib
volumentations-3D
pillow
scipy
tqdm
```


## 下載

請下載最新版，將訓練過的權重移至 checkpoints/，資料集解壓至根目錄命名為 dataset

* [Datasets](https://drive.google.com/drive/folders/1-JzW3uJlVD9JAbnb1OOzNJoTU9AFqbVK?usp=share_link)
* [Weights](https://drive.google.com/drive/folders/1p-2f3j27dXx-ZGQwakJ1FI1xxmWi3ox5?usp=share_link)


## 測試

```bash
python predict.py
```


## 評估

|            | Data | Accuracy | Loss |
| ---------- | ---- | -------- | -----|
| Train      | ??   | ??       | ??   |
| Validation | ??   | ??       | ??   |

## 模型訓練

訓練前確認是否有以下資料夾及檔案

```bash
dataset/
├───HIGH/
│   ├───1182214(O)/
│   │       .
│   │       .
│   └───1182214(O)/
│
└───LOW/
    ├───0708758(O)/
    │       .
    │       .
    └───0708758(O)/       
```


## 方法


### YOLOv5 Object Detection

* 物件偵測使用 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 來做訓練，框出心臟的位置，排除切片中多餘的器官及黑色區域

* 取所有切片中 bounding box 的最左上、最右下座標，作為裁切影像的依據，保證可以框住整顆 3D 心臟

<p float="left">
  <img src="doc/yolov5_1.jpg" width="400" />
</p>
<p float="left">
  <img src="doc/yolov5_2.jpg" width="400" />
</p>

### Uniformizing Techniques

### 3D-CNN Model

### Guided-GradCAM

## 遭遇問題及困難

* ...
* ...
* ...



## 預計增加功能

* [ ] ...
* [ ] ..
* [ ] ...
* [ ] ...
  