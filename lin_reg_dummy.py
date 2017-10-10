import tensorflow as tf
import numpy as np

# creating dummy data
X = np.linspace(-1,1,100)
Y = 3*X + np.random.randn(X.shape[0])

# weights and biases
w = tf.Variable(0.0, name = "Weight_1")
b = tf.Variable(0.0, name = "bias")

# model
Y_pred = X*w +b

#optimization
loss = tf.square(Y-Y_pred, name = "loss")

optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for x,y in zip(X,Y):
            sess.run(optimizer)


    w_final, b_final = sess.run([w,b])
print(w_final,b_final)