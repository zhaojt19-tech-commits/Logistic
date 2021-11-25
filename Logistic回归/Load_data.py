import scipy.io as scio
import random
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# 实现sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def shuffle(X, Y, seed=0):
    """
    随机打乱原始数据
    """
    random.seed(0)
    index = [i for i in range(X.shape[0])]
    random.shuffle(index)
    return X[index], Y[index]


def get_zero_and_one(X, Y):
    """
    从给定数据中抽取数字0和数字1的样本
    """
    index_1 = Y==0
    index_8 = Y==1
    index = index_1 + index_8
    return X[index], Y[index]
    

def load_data(data_dir="./", data_file="mnist.mat"):
    # 加载数据，划分数据集
    data = scio.loadmat(os.path.join(data_dir, data_file))
    train_X, test_X = data['train_X'], data['test_X']
    train_Y, test_Y = data['train_Y'].reshape(train_X.shape[0]), data['test_Y'].reshape(test_X.shape[0])
    
    # 从训练数据中抽取数字 0 和数字 1 的样本，并打乱
    train_X, train_Y = get_zero_and_one(train_X, train_Y)
    train_X, train_Y = shuffle(train_X, train_Y)
    train_Y = (train_Y==1).astype(np.float32)  # 1->True 0->false,将train_Y转化为bool数组
    # 从测试数据中抽取数字 0 和数字 1 的样本，并打乱
    test_X, test_Y = get_zero_and_one(test_X, test_Y)
    test_X, test_Y = shuffle(test_X, test_Y)
    test_Y = (test_Y==1).astype(np.float32)    # 2->True 0->false,将test_Y转化为bool数组
    print("原始图片共有%d张，其中数字1的图片有%d张。" % (test_X.shape[0], sum(test_Y==1)))
    return train_X, train_Y, test_X, test_Y
    
    
def ext_feature(train_X, test_X):
    """
    抽取图像的白色像素点比例作为特征
    """
    train_feature = np.sum(train_X>200, axis=1)/784
    test_feature = np.sum(test_X>200, axis=1)/784
    return train_feature, test_feature

def log_likelihood(X, Y, w, b):
    # 首先按照标签来提取正样本和负样本的下标
    pos, neg = np.where(Y == True), np.where(Y == False)

    # 对于正样本计算 loss, 这里我们使用了matrix operation。 如果把每一个样本都循环一遍效率会很低。
    pos_sum = np.sum(np.log(sigmoid(np.dot(X[pos], w) + b)))

    # 对于负样本计算 loss
    neg_sum = np.sum(np.log(1 - sigmoid(np.dot(X[neg], w) + b)))

    # 返回cross entropy loss
    return -(pos_sum + neg_sum)

def train(w, b, X, Y, alpha=0.1, epochs=50, batchsize=32):
    print("随机梯度下降后的交叉熵损失：")
    for step in range(epochs):
        #随机采样一个batch, batch大小为32
        idx = np.random.choice(X.shape[0], batchsize)
        batch_x = X[idx]
        batch_Y = Y[idx]

        #计算预测值与实际值之间的误差
        error = sigmoid(np.dot(batch_x, w) + b) - batch_Y

        #对于w, b的梯度计算
        grad_w = np.matmul(batch_x.T, error)
        grad_b = np.sum(error)

        # 对于w, b的梯度更新
        w = w - alpha * grad_w
        b = b - alpha * grad_b

        # TODO 查看交叉熵损失的变化情况（可以删除）
        print(step, log_likelihood(X, Y, w, b))

    return w, b


def test(w, b, X, Y):
    #计算X中每个样本的概率值
    h = sigmoid(np.dot(X, w) + b)

    #预测结果（true/False）
    result_pre = (h > 0.5).astype(np.float32)
    compare = (result_pre == Y).astype(np.float32)
    correct = np.sum(compare)

    #计算正确率
    rate_correct = correct / Y.shape[0]

    #计算真阳性率
    temp = result_pre + Y
    temp = (temp == 2).astype(np.float32)
    TP = np.sum(temp)
    TRP = TP / np.sum(Y)
    return rate_correct, TRP



if __name__ == "__main__":
    # 加载数据
    train_X, train_Y, test_X, test_Y = load_data(data_dir="C:\Project\Python Project\Logistic回归")
    # 抽取特征
    train_feature, test_feature = ext_feature(train_X, test_X)
    
    # 随机初始化参数
    w = np.random.randn()
    b = np.random.randn()

    #利用自己的logistic回归算法训练w_t,b_t
    w_t, b_t = train(w, b, train_feature, train_Y, 0.1, 50, 32)   #echos参数可以改得大一点，实际上每一个echos都应该进行多个batch的梯度下降直到这些batch的数量达到训练集样本数

    print("自己写的逻辑回归的参数w, b分别为: ", w_t, b_t,)

    # 测试
    rate_correct, TRP = test(w_t, b_t, test_feature, test_Y)
    print("自己写的部分的测试正确率为：")
    print(rate_correct)
    print("自己写的部分的真阳性率为：")
    print(TRP)

    # 调用sklearn的模块来训练，看看跟自己的结果区别。
    # C设置一个很大的值，意味着不想加入正则项 （这里就理解成为了公平的比较）
    clf = LogisticRegression(fit_intercept=True, C=1e15)
    temp = train_feature.shape[0]
    clf.fit(train_feature.reshape(temp, 1), train_Y)
    w_s = clf.coef_[0][0]
    b_s = clf.intercept_[0]
    print("(sklearn)逻辑回归的参数w, b分别为: ", w_s, b_s, )

    # 测试
    rate_correct = test(w_s, b_s, test_feature, test_Y)
    print("(sklearn)逻辑回归的测试正确率为：")
    print(rate_correct)

    