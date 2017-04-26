# -*- coding:utf-8 -*-
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib as mpl

# 初始化变量设置
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in+fan_out))
    high = constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low,maxval=high,dtype=tf.float32)

np.random.seed(2)

# 定义去噪自编码的类
class Additive_Gaussian_Noise_Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):

        # 设置了框架参数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义网络结构
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input, )), self.weights['w1']), self.weights['b1']))     #
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 计算损失函数
        self.cost = 0.5*tf.reduce_sum((tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)))
        self.optimizer = optimizer.minimize(loss=self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 参数初始化函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    # 执行一步训练，返回当前cost
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),feed_dict={self.x: X, self.scale:self.training_scale})
        return cost

    # 测试，计算cost，不触发训练操作
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale:self.training_scale})

    # transform函数，用于返回隐含层
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale:self.training_scale})

    # generate函数，输入隐含层，将高阶数据复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstrction, feed_dict={self.hidden:hidden})

    # reconstruct函数，整体运行一遍，输入原始数据，输出复原数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale:self.training_scale})

    # 获取隐含层的权重w1
    def get_weights(self):
        return self.sess.run(self.wights['w1'])

    # 获取隐含层偏置系数b1
    def get_biases(self):
        return self.sess.run(self.wights['b1'])

# 载入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# 标准化处理
def standard_scale(x_train, x_test):
    prepocessor = prep.StandardScaler().fit(x_train)
    x_train = prepocessor.transform(x_train)
    x_test = prepocessor.transform(x_test)
    return x_train,x_test

# 获取随机block数据
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# 操作
# 标准化变换
x_train, x_test = standard_scale(mnist.train.images, mnist.test.images)
n_sample = int(mnist.train.num_examples) #即等效于这样？x_train.shape[0]
training_epochs = 20
batch_size = 128
display_step = 1

# 创建实例
autoencoder = Additive_Gaussian_Noise_Autoencoder(n_input=784,
                                                  n_hidden=200,
                                                  transfer_function=tf.nn.softplus,
                                                  optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                  scale=0.01)

# 训练
# cost_plot = []
avg_cost_plot = []

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_sample / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(x_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_sample*batch_size

    avg_cost_plot.append(avg_cost)

    if epoch % display_step == 0:
        print("Epoch: ", '%04d'%(epoch+1), "cost= ", "{:.9f}".format(avg_cost))

print("Total cost: "+str(autoencoder.calc_total_cost(x_test)))

# 绘制cost曲线
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.plot(avg_cost_plot,color='r')
plt.xlabel(u'迭代次数', fontsize=12)
plt.ylabel(u'均方差（MSE）', fontsize=12)
plt.title('avg_cost_plot', fontsize=14)
plt.show()













