import os
import tensorflow.compat.v1 as tf
import pickle

def write_graph():
    """
    """
    tf.disable_v2_behavior()
    inp = tf.placeholder(tf.float32, shape=(None, 4), name="inputs")
    constant_input = tf.placeholder(tf.int32, name="const_input")
    W = tf.Variable(tf.random_normal((4, 3)), name="W")
    b = tf.Variable(tf.zeros(1))
    logits = tf.identity(tf.matmul(inp, W) + b, name="logits")
    y_pred = tf.nn.softmax(logits, name="probas")

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        os.makedirs("model_files/", exist_ok=True)
        saver.save(sess, "model_files/model.ckpt")
    # write the constant input
    with open("model_files/const_feed_dict.pkl", "wb") as f:
        pickle.dump({"const_input": 10}, f, protocol=2)


if __name__ == "__main__":
    write_graph()
