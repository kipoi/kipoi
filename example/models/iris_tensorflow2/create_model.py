import tensorflow as tf
import numpy as np

class BasicModel(tf.Module):
  def __init__(self):
    super().__init__()
    self.bias = tf.Variable(1.)
    self.weight = tf.Variable(tf.random.normal((4, 3))) 

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, inputs):
    logits = tf.identity(tf.matmul(inputs, self.weight) + self.bias)
    y_pred = tf.nn.softmax(logits, name="probas")
    return y_pred

if __name__ == "__main__":
    model = BasicModel()
    tf.saved_model.save(model, "model_files")
    reconstructed_model = tf.saved_model.load("model_files")
    test_input = np.ones((3, 4))
    np.testing.assert_allclose(model(test_input), reconstructed_model(test_input))
