# this script test behavior of customized Conv121 layer

from layers.Conv2D121 import Conv2D121
from keras.models import Sequential
from keras.layers import Activation, Conv2D
import random
import numpy as np
import tensorflow as tf

random.seed(0)

model = Sequential()
conv1 = Conv2D121(3, (2, 2), padding='same',
                  input_shape=(4, 4, 3), use_bias=True)
model.add(conv1)
model.add(Activation('relu'))

input_ = [
    [[[1,2,3], [4,5,6], [7,8,9], [10,11,12]],
    [[13,14,15], [16,17,18], [19,20,21], [22,23,24]],
    [[25,26,27], [28,29,30], [31,32,33], [34,35,36]],
    [[37,38,39], [40,41,42], [43,44,45], [46,47,48]]]
]
input_value = np.array(input_)

weights = conv1.get_weights()

init = tf.global_variables_initializer()
input_x = tf.placeholder(tf.float32, [None, 4, 4, 3], name='input_x')
output_tensor = conv1(input_x)

sess = tf.Session()

sess.run(init)
print("input:")
print(input_value)
print("")
print("weight:")
print(weights)
print("")
print("output:")
print(sess.run(output_tensor, feed_dict={
    input_x: input_value
}))

sess.close()

print("done")