import tensorflow as tf

# Step 1: reset graph
tf.reset_default_graph()

# Step 2: make placeholders or variables
test_constant = tf.constant(10.0, dtype=tf.float32)


# Step 3: do operations on placeholders or variables
add_one_operation = test_constant + 1


# Step 4: run a tensorflow session
with tf.Session() as sess:
    print(sess.run(add_one_operation))




