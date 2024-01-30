Python 机器学习工具库 Scikit-Learn ，建立在 NumPy、SciPy、Pandas 和 Matplotlib 之上，是最常用的 Python 机器学习工具库之一
# SKLearn是什么
Scikit-Learn，简称SKLearn
封装了常用的机器学习方法，包括分类、回归、聚类、降维、模型评估、数据预处理等
官网：https://scikit-learn.org/stable/

# SKLearn常用接口
## 数据集导入
SKLearn官网：https://scikit-learn.org/stable/modules/classes.html?highlight=dataset#module-sklearn.datasets

```python
#鸢尾花数据集
from sklearn.datasets import load_iris
#乳腺癌数据集
from sklearn.datasets import load_breast_cancer
#波士顿房价数据集
from sklearn.datasets import load_boston
```
## 数据预处理
官网链接：https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

```python
#拆分数据集
from sklearn.model_selection import train_test_split
#数据缩放
from sklearn.preprocessing import MinMaxScaler
```

## 特征抽取
官网链接：https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction

```python
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo':1, 'bar':2}, {'foo':3, 'baz':1}]
X = v.fit_transform(D)
```

## 特征选择
官网链接：https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
X, y = load_digits(return_X_y = True)
## 特征选择
X_new = SelectKBest(chi2, k=20).fit_transform(X,y)
```

## 常用模型
官网链接：https://scikit-learn.org/stable/modules/classes.html

```python
## KNN模型
from sklearn.neighbors import KNeighborsClassifier
## 决策树
from sklearn.tree import DecisionTreeClassifier
## 支持向量机
from sklearn.svm import SVC
## 随机森林
from sklearn.ensemble import RandomForestClassifier
```

## 建模拟合与预测
```python
# 拟合训练集
knn.fit(X_train, y_train)
# 预测
y_pred = knn.predict(X_test)
```

## 模型评估
官网链接：https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

```python
# 求精度
knn.score(X_test, y_test)

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix

# 绘制ROC曲线
from sklearn.metrics import roc_curve, roc_auc_score
```

## 典型的建模流程示例 
