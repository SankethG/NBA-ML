import tensorflow as tf
import matplotlib.pyplot as plt

# Resetting graph
tf.reset_default_graph()

#define variables and constants
input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)


slope = tf.Variable(0.5, dtype=tf.float32)
intercept = tf.Variable(0.1, dtype=tf.float32)

#operations

model_operation = slope * input_data + intercept
error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.010)
train = optimizer.minimize(loss)

# Run tensorflow

init = tf.global_variables_initializer()

# User input

x_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
y_values=[38, 33, 14, 46, 34, 40, 31.5, 22, 20.5, 27.4, 47.5, 48, 20.5, 20 ]

with tf.Session() as sess:
    sess.run(init)
    for i in range(0, 2000):
        sess.run(train, feed_dict = {input_data: x_values, output_data: y_values})
        if i % 100 == 0:
            print(sess.run([slope, intercept]))
            plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))

    print(sess.run(loss, feed_dict={input_data: x_values, output_data: y_values}))
    plt.plot(x_values, y_values, 'ro', 'Training Data')
    plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))
    plt.show()