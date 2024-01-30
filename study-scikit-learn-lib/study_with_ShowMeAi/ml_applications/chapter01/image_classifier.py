## 导入工具库
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

# 提取图像的像素特征
def extract_color_stats(image):
    '''
    将图片分成 RGB 三通道，然后分别计算每个通道的均值和标准差，然后返回
    :param image:
    :return:
    '''
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
    return features

## 查看当前文件路径
print(os.getcwd())

## 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())

## 定义一个保存模型的字典，根据 key 来选择加载哪个模型
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=20000),
    "svm": SVC(kernel="rbf", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

## 加载数据并提取特征
print("抽取图像特征中...")
# print(args['dataset'])
imagePaths = list(paths.list_images(args['dataset']))
# print("一共有", len(imagePaths), "张图片")
# print("图片路径示例", imagePaths[0])
data = []
labels = []

## 循环遍历所有的图片数据
for imagePath in imagePaths:
    # 加载图片，然后计算图片的颜色通道统计信息
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    # 保存图片的标签信息
    # 使用 os.path.basename 获取文件名
    filename = os.path.basename(imagePath)
    # 使用 os.path.splitext 分离文件名和扩展名
    filename_without_extension, _ = os.path.splitext(filename)
    # 使用 split 获取目录中的标签信息
    label = filename_without_extension.split('_')[-2]
    # label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# print("展示data数据...")
# for i in range(5):
#     print(data[i])
# print("展示label标签...")
# for i in range(5):
#     print(labels[i])

## 对标签进行编码，从字符串变为整数类型
le = LabelEncoder()
labels = le.fit_transform(labels)

# print("展示编码后的label标签...")
# for i in range(5):
#     print(labels[500+i])

## 进行训练集和测试集的划分，80%数据作为训练集，其余20%作为测试集
trainX, testX, trainY, testY = train_test_split(data, labels, random_state=3, test_size=0.2)
# print('trainX numbers={}, testX numbers={}'.format(len(trainX), len(testX)))

## 训练模型
print("[应用 '{}' 模型建模".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

## 预测并输出分类结果报告
print("模型评估")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))
