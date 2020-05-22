import tensorflow as tf 

t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])


a = tf.constant([[1, 2, 3]])
print(a.shape)

my_slice = tf.slice(a, [0, 2], [1, 1])
print(my_slice)