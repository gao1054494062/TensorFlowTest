import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[4,3]])
matrix2 = tf.constant([[1],[2]])

product = tf.matmul(matrix1,matrix2)
#np.dot(m1,m2)

#method 1
sess1 = tf.Session()
result = sess1.run(product)
print("result 1 = ",result)
sess1.close()

#method 2
with tf.Session() as sess1:
    result = sess1.run(product)
    print("result 2 = ",result)

