from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file

# 加载鸢尾花数据集
iris = load_iris()

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 将训练集转换为 libsvm 格式
dump_svmlight_file(X_train, y_train, 'train.libsvm', zero_based=True)

# 将测试集转换为 libsvm 格式
dump_svmlight_file(X_test, y_test, 'test.libsvm', zero_based=True)