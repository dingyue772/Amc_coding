"""
使用问题：keras的版本与源代码有出入，需要修改部分代码，包括
将原代码中的from keras.layers.convolutional import Conv2D修改为from keras.layers import Conv2D
将原代码中的from keras.optimizers import Adam修改为from tensorflow.keras.optimizers import legacy
"""

## 导入工具库
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.optimizers import legacy
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

## 配置参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
args = vars(ap.parse_args())

## 加载数据并提取特征
print("抽取图像特征中...")
imagePaths = paths.list_images(args['dataset'])
data = []
labels = []

## 循环遍历所有的图片数据
for imagePath in imagePaths:
    # 加载图片，然后调整成 32×32 大小，并做归一化到 [0,1]
    image = Image.open(imagePath)
    image = np.array(image.resize((32, 32))) / 255.0
    data.append(image)
    # 保存图片的标签信息
    # 使用 os.path.basename 获取文件名
    filename = os.path.basename(imagePath)
    # 使用 os.path.splitext 分离文件名和扩展名
    filename_without_extension, _ = os.path.splitext(filename)
    # 使用 split 获取目录中的标签信息
    label = filename_without_extension.split('_')[-2]
    # label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

## 对标签编码，从字符串变为整型
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

## 划分训练集和测试集
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

## 定义 CNN 网络模型结构
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation("softmax"))

## 训练模型
print("训练网络中...")
opt = legacy.Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=50, batch_size=32)

## 预测
print("评估模型效果...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))


    