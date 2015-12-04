'''
layer types that have tied weights with another layer
'''


from ..layers.core import Layer, Dense
from .. import activations, initializations, regularizers, constraints
from ..layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from ..layers import containers

def transpose(layer):
    '''
    Returns the transposed version of a layer
    Dense(W, bv) -> Dense(W.T, bh)
    Conv2D(W, bv) -> Corr2D(W, bv)
    '''
    if type(layer) is list:
        try:
            return [transpose(l) for l in layer]
        except Exception as e:
            raise e
    if type(layer) is Dense:
        return TiedDense(transposed=layer)
    # TODO add support for more layers
    if type(layer) is Convolution2D:
        return TiedConvolution2D(layer)
    if type(layer) is MaxPooling2D:
        return UpSampling2D(size=layer.pool_size, dim_ordering=layer.dim_ordering)
    raise "Layer not supported for use in boltzman machines"

class TiedConvolution2D(Layer):
    input_ndim = 4

    def __init__(self, transposed, subsample=(1,1),
                 b_regularizer=None, activity_regularizer=None,
                 b_constraint=None, **kwargs):

        assert type(transposed) is Convolution2D

        self.transposed = transposed
        self.activation = transposed.activation
        self.subsample = tuple(subsample)

        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.b_constraint]

        super(TiedConvolution2D, self).__init__(**kwargs)

    def build(self):
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.b]
        self.regularizers = []

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

    @property
    def dim_ordering(self):
        return self.transposed.dim_ordering

    @property
    def nb_row(self):
        return self.transposed.nb_row

    @property
    def nb_col(self):
        return self.transposed.nb_col

    @property
    def border_mode(self):
        return border_transp[self.transposed.border_mode]

    @property
    def nb_filter(self):
        return self.transposed.input_shape[1]

    @property
    def input_shape(self):
        return self.transposed.output_shape

    def set_weights(self, weights):
        raise NotImplementedError("Not supported because of weight tieing")

    @property
    def W_shape(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            return (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            return (self.nb_row, self.nb_col, stack_size, self.nb_filter)

    @property
    def W(self):
        return K.permute_dimensions(self.transposed.W[:, :, ::-1, ::-1], (1, 0, 2, 3))

    @property
    def output_shape(self):
        return self.transposed.input_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering)

        output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        output = self.activation(output)
        return output


class TiedDense(Layer):
    input_ndim = 2
    def __init__(self, init='glorot_uniform', activation='linear',
                 b_regularizer=None, activity_regularizer=None,
                 b_constraint=None, transposed=None, **kwargs):
        self.init = initializations.get(init)
        self.transposed = transposed
        self.activation = activations.get(activation)
        self.output_dim = self.transposed._input_shape[1]

        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.b_constraint]

        kwargs['input_shape'] = (self.transposed.output_dim,)
        print(self.output_dim, kwargs['input_shape'])
        super(TiedDense, self).__init__(**kwargs)

    def build(self):
        self.input = T.matrix()
        self.b = shared_zeros((self.output_dim,))

        self.params = [self.b]

        self.regularizers = []

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

    def set_weights(self, weights):
        raise NotImplementedError("Not supported because of weight tieing")

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.transposed.W.T) + self.b)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim}
        base_config = super(TiedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
