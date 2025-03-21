import tensorflow as tf
print(tf.__version__)

@tf.function
def test_print():
    tf.print(tf.__version__)
    start = tf.timestamp()
    end = tf.timestamp()
    diff = end - start
    tf.print("Time: ", tf.strings.format("{:.2f}", diff), " seconds")

test_print()