import pandas as pd
import tensorflow as tf

# read data
data = pd.read_excel(open('fire_theft_chicago/slr05.xls','r'), error_bad_lines = False, delimiter ='\t')
num_ep = 4


# create placeholders
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

#initialize weights and biases
w = tf.Variable(0.0, name = "weights_1")
u = tf.Variable(0.0,name = "weights_2")
b = tf.Variable(0.0, name = "bias")

# the model
#Y_pred = X*w + b # straight line fitting
Y_pred = X*X*w +X*u + b # quadrilateral fitting


# optimization
loss = tf.square(Y-Y_pred,name = "loss")

optimizer = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #training the model
    for i in range(num_ep):
        for x,y in zip(data['X'],data['Y']):
            sess.run(optimizer,feed_dict={X:x,Y:y})

    w_fin, u_fin,b_fin = sess.run([w,u,b])
print(w_fin, u_fin, b_fin)