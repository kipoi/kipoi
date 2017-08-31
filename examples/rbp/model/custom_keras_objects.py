from concise.layers import SplineT

OBJECTS = {"SplineT": SplineT}
# TODO - register special layers

# from keras.engine.topology import Layer
# from keras import regularizers
# from keras import initializers
# from keras.engine import InputSpec
# import keras.backend as K


# class SplineT(Layer):
#     """Spline transformation layer.

#     As input, it needs an array of scalars pre-processed by `concise.preprocessing.EncodeSplines`
#     Specifically, the input/output dimensions are:

#     - Input: N x ... x channels x n_bases
#     - Output: N x ... x channels

#     # Arguments
#         shared_weights: bool, if True spline transformation weights
#     are shared across different features
#         kernel_regularizer: use `concise.regularizers.SplineSmoother`
#         other arguments: See `keras.layers.Dense`
#     """

#     def __init__(self,
#                  # regularization
#                  shared_weights=False,
#                  kernel_regularizer=None,
#                  use_bias=False,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  **kwargs
#                  ):
#         super(SplineT, self).__init__(**kwargs)  # Be sure to call this somewhere!

#         self.shared_weights = shared_weights
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)

#         self.input_spec = InputSpec(min_ndim=3)

#     def build(self, input_shape):
#         assert len(input_shape) >= 3

#         n_bases = input_shape[-1]
#         n_features = input_shape[-2]

#         # self.input_shape = input_shape
#         self.inp_shape = input_shape
#         self.n_features = n_features
#         self.n_bases = n_bases

#         if self.shared_weights:
#             use_n_features = 1
#         else:
#             use_n_features = self.n_features

#         # print("n_bases: {0}".format(n_bases))
#         # print("n_features: {0}".format(n_features))

#         self.kernel = self.add_weight(shape=(n_bases, use_n_features),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       trainable=True)

#         if self.use_bias:
#             self.bias = self.add_weight((n_features, ),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=None)

#         self.built = True
#         super(SplineT, self).build(input_shape)  # Be sure to call this somewhere!

#     def compute_output_shape(self, input_shape):
#         return input_shape[:-1]

#     def call(self, inputs):
#         N = len(self.inp_shape)
#         # put -2 axis (features) to the front
#         # import pdb
#         # pdb.set_trace()

#         if self.shared_weights:
#             return K.squeeze(K.dot(inputs, self.kernel), -1)

#         output = K.permute_dimensions(inputs, (N - 2, ) + tuple(range(N - 2)) + (N - 1,))

#         output_reshaped = K.reshape(output, (self.n_features, -1, self.n_bases))
#         bd_output = K.batch_dot(output_reshaped, K.transpose(self.kernel))
#         output = K.reshape(bd_output, (self.n_features, -1) + self.inp_shape[1:(N - 2)])
#         # move axis 0 (features) to back
#         output = K.permute_dimensions(output, tuple(range(1, N - 1)) + (0,))
#         if self.use_bias:
#             output = K.bias_add(output, self.bias, data_format="channels_last")
#         return output

#     def get_config(self):
#         config = {
#             'shared_weights': self.shared_weights,
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer)
#         }
#         base_config = super(SplineT, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
