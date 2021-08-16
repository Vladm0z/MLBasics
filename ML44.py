import tensorflow as tf 

with tf.compat.v1.Session() as sess:
  x1 = tf.constant(5)
  x2 = tf.constant(6)
  result = tf.multiply(x1, x2)
  
  print(sess.run(result))

