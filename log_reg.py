import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Read in data
MNIST = input_data.read_data_sets('MNIST_data/',one_hot = True)

#specify parameters
learning_rate =0.01
batch_size = 128
n_epoch = 25

# placeholders for input and label
X = tf.placeholder(tf.float32,[batch_size ,784],name = "image")
Y = tf.placeholder(tf.float32, [batch_size ,10],name = "label")

# model specification
w = tf.Variable(tf.random_normal(shape = [784,10], stddev =0.01),name = "weights")
b = tf.Variable(tf.zeros([1,10]),name = "bias")

Y_pred = tf.matmul(X,w) + b

#define the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred,labels =Y)
loss = tf.reduce_mean(entropy)

#optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = MNIST.train.num_examples/batch_size
    for e in range(n_epoch):
        for _ in range(n_batches):
            X_batch,Y_batch = MNIST.train.next_batch(batch_size)
            sess.run([optimizer,loss], feed_dict={X:X_batch, Y:Y_batch})
            #print (sess.run(loss,feed_dict= {X:X_batch,Y:Y_batch}))
