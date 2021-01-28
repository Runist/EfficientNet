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
train_dir = r'D:\Python_Code\BasicNet\dataset\train'		# chang to your dataset directory
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

