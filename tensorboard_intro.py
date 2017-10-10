import tensorflow as tf

a = tf.constant([3,4], name= 'a')
b = tf.constant([0,2], name = 'b')

c = tf.add(a,b, name= 'c')

with tf.Session() as sess:
    print (sess.run(c))
    writer = tf.summary.FileWriter('graph_add', sess.graph)

writer.close()