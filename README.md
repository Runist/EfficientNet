# EfficientNet

> EfficientNet implement by tf2 <br>

## 1. Quick Start.

1. Clone the repository.

```
$ git clone https://github.com/Runist/EfficientNet.git
```

2. You are supposed to install some dependencies before getting out hands with these codes.

```
$ cd EfficientNet
$ pip install -r requirements.txt
```

3. Download flower dataset.

```
$ wget https://github.com/Runist/BasicNet/releases/download/1.0/dataset.rar
```

## 2. Train your dataset.

Something you need to change:

- train.py

```python
train_dir = r'D:\Python_Code\BasicNet\dataset\train'		# change to your dataset directory
val_dir = r'D:\Python_Code\BasicNet\dataset\validation'
```

Finally

```
$ python train.py			
```

## 3. Show your predict result.

Something you need to change:

- predict.py

```python
img_path = r'Your test image path.'
```

And then

```
$ python predict.py
```

## 4. See the features you learned from the network.

Again, you need to modify the path of image:

- gard_cam.py

```python
img_path = r'Your test image path.'
```

After that

```
$ python predict.py
```

You will get a heat map. and the redder the area, the more prominent the feature.

![heat_map.jpg](https://i.loli.net/2021/02/01/thlB5uQPFxAndOE.jpg)