#author: https://medium.com/@curiousily/tensorflow-for-hackers-part-i-basics-2c46bc99c930

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Y = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.05))(Y)
a_var = tf.Variable(1.0)
b_var = tf.Variable(1.0)
y_var = a_var * X + b_var
plt.plot(X, Y);

loss = tf.reduce_mean(tf.square(y_var - Y))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

TRAINING_STEPS = 300
results = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(TRAINING_STEPS):
        results.append(sess.run([train, a_var, b_var])[1:])
        
final_pred = results[-1]
a_hat = final_pred[0]
b_hat = final_pred[1]
y_hat = a_hat * X + b_hat

print("a:", a_hat, "b:", b_hat)
plt.plot(X, Y);
plt.plot(X, y_hat);
