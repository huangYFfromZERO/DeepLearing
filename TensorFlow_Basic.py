# DeepLearing
import tensorflow as tf
import numpy as np
#对于dot的一个小实验
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([[0.2,0.1]],x_data) +0.300
z_data = np.dot([0.2,0.1],x_data)+0.300
#print(y_data.shape)
#print(z_data.shape)
#所以这里可以看出对于dot函数而言，如果没有明确的给出ndarry的情况，是不会严格按照矩阵运算来进行处理的
#线性神经网络模型
#偏置量起初都是0，权值为随机给定的
b = tf.Variable(tf.zeros([1,1]))              #实际上也只是根据规定的ndarray
#print(b.shape)
w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0),dtype=tf.float32)
y = tf.matmul(w,x_data)+b
#得到了训练出来的一百个y
loss = tf.reduce_mean(tf.square(y - y_data))
#求出100个y的平均lost
optimizer = tf.train.GradientDescentOptimizer(0.5)               #采用梯度下降算法，其中参数为学习率
#记得之前我所写的是利用sigmoid函数求出每次微小的变化量，再计算出每层需要改变多少，再进行代入计算去进行更新的
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(0, 201):
    sess.run(train)
    #if step % 20 == 0:
    print (step, sess.run(w), sess.run(b))
