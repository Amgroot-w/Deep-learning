
import numpy as np
import pandas as pd
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data
import time

start = time.time()
# 导入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
np.random.shuffle(mnist.train)  # 洗牌
np.random.shuffle(mnist.test)  # 洗牌

# 训练
clf = svm.SVC(kernel='rbf', C=0.3, gamma=70)
train_num = 5000
print('\nSVM拟合中...')
clf.fit(mnist.train.images[:train_num, :], mnist.train.labels[:train_num])
print('拟合完成！')

# 测试
test_num = 5000
pred = clf.predict(mnist.test.images[:test_num, :])
ac = np.mean(np.equal(pred, mnist.test.labels))

end = time.time()
print('用时：%.2f秒' % (end-start))















