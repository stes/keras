from ..layers.core import Layer, Dense
from .. import activations, initializations, regularizers, constraints
from ..layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from ..layers import containers

from regularizers import SparsityRegularizer
import samplers

import keras.backend as K

border_transp = {'same': 'same', 'valid': 'full', 'full': 'valid'}

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

class BoltzmanMachine(Layer):
    
    def __init__(self, layer, vis_sampler, hid_sampler, free_energy,
            mode, output_reconstruction=True, weights=None,**kwargs):
        self.input = T.matrix()
        super(BoltzmanMachine, self).__init__(**kwargs)
        self.output_reconstruction = output_reconstruction
        self.propup_layer = layer
        self.propdown_layer = transpose(layer)
        self.vis_sampler = samplers.get(vis_sampler)
        self.hid_sampler = samplers.get(hid_sampler)
        self.layers = [self.propup_layer, self.propdown_layer]
        self.persistent = None
        self.propup_activation = activations.get('sigmoid')
        self.propdown_activation = activations.get('sigmoid')
        self.params = []
        self.mode = mode
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for layer in self.layers:#.get_param():
            params, reg, constr, updates = layer.get_params()
            self.regularizers += reg
            self.updates += updates
            for p, c in zip(params, constr):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)
        if weights is not None:
            self.set_weights(weights)
        self.propup_layer.b -= 1.

    def set_previous(self, node):
        self.previous = node

    def get_weights(self):
        return [self.propup_layer.get_weights(),
                self.propdown_layer.get_weights()]

    def set_weights(self, weights):
        for layer, weight in zip(self.layers, weights):
            layer.set_weights(weight)

    @property
    def encoder(self):
        return self.popup_layer

    @property
    def decoder(self):
        return self.propdown_layer

    @property
    def input_shape(self):
        return self.propup_layer.input_shape

# some wrapper function to access the rbm
    @property
    def bv(self):
        return self.propdown_layer.b

    @property
    def bh(self):
        return self.propup_layer.b

    @property
    def W(self):
        return self.propup_layer.W

    @property
    def output_shape(self):
        if self.train:
            return self.propdown_layer.previous.output_shape
        if self.output_reconstruction:
            return self.propup_layer.previous.output_shape
        else:
            return self.propup_layer.previous.output_shape

    def get_output(self, train=False):
        x = self.gibbs_vhv(self.get_input(train))[-1]
        return theano.gradient.disconnected_grad(x)
        print ('mode', self.mode)
        if train:
            _,_,x = self.gibbs_vhv(self.get_input(train))[-1]
            return theano.gradient.disconnected_grad(x)
        if not train:
            return self.gibbs_mcmc(1)[-1]
            if self.mode == 'inference sampling':
                return self.propdown_layer.get_output(train)
            if self.mode == 'inference mean':
                return self.propdown_layer.get_output(train)
            if self.mode == 'features':
                return self.propup_layer.get_output(train)
        raise ValueError

    def get_config(self):
        # TODO
        return {"name": self.__class__.__name__,
                "output_reconstruction": self.output_reconstruction}

    def to_autoencoder(self):
        # TODO
        pass

    def free_energy(self, nb_steps):
        if nb_steps is 0:
            return self._free_energy(self.input)
        psigm, pmean, psample = self.gibbs_mcmc(nb_steps)
        return self._free_energy(psample)

    def cdk_loss(self):
        def loss(y_true, y_pred):
            return T.mean(self._free_energy(y_true)) - T.mean(self._free_energy(y_pred))
        return loss
    
    def _free_energy(self, x):
        return self._free_energy_bernoulli(x)
    
    def _free_energy_gaussian(self, x):
        # gaussian visibles, assumes sigma=1.
        wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
        vbias_term = 0.5*T.sum((x - self.propdown_layer.b)**2, axis=1)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term + vbias_term

    def _free_energy_bernoulli(self, x):
        # bernoulli visibles
        wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
        vbias_term = T.dot(x, self.propdown_layer.b)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term   

    def get_bernoulli_F(self):
        def F(x):
            # bernoulli visibles
            wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
            vbias_term = T.dot(x, self.propdown_layer.b)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term - vbias_term   
        return F

    def gibbs_mcmc(self, nb_gibbs_steps=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.get_input(train=True))
        chain_start = ph_sample
        #chain_start = self.persistent
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=nb_gibbs_steps
        )
        chain_end = theano.gradient.disconnected_grad(nv_samples[-1])
        return pre_sigmoid_nvs[-1], nv_means[-1], chain_end
 
    def sample_h_given_v(self, v0_sample):
        self.propup_layer.input = v0_sample
        pre_sigmoid_h1 = self.propup_layer.get_output(v0_sample)
        h1_mean = self.propup_activation(pre_sigmoid_h1)
        h1_sample = self.hid_sampler(h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        self.propdown_layer.input = h0_sample
        pre_sigmoid_v1 = self.propdown_layer.get_output(h0_sample)
        v1_mean = pre_sigmoid_v1 #self.propdown_activation(pre_sigmoid_v1)
        v1_sample = pre_sigmoid_v1 #self.vis_sampler(v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]



def free_energy_score_matching(self, sigma):
#taken from pylearn2 example.
#See https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/energy_functions/rbm_energy.py
    def F(V):
        assert V.ndim == 2

        bias_term = T.dot(V,self.bv)
        bias_term.name = 'bias_term'
        assert len(bias_term.type.broadcastable) == 1

        sq_term = 0.5 * T.sqr(V).sum(axis=1)
        sq_term.name = 'sq_term'
        assert len(sq_term.type.broadcastable) == 1

        softplus_term =  T.nnet.softplus( (self.transformer.lmul(V)+self.bias_hid) / T.sqr(self.sigma)).sum(axis=1)
        assert len(softplus_term.type.broadcastable) == 1
        softplus_term.name = 'softplus_term'

        return (
                sq_term
                - bias_term
                ) / T.sqr(sigma) - softplus_term
    return F

class RBMUnits():
    
    def __init__(self, rbm):
        self.rbm = rbm

    def activation(self, x):
        def act(x):
            return x
        return act

    def sample(self, x):
        def sample(x):
            return x
        return sample

class BernoulliUnits():

    def free_energy(self, x):
        def F(x):
            wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
            vbias_term = T.dot(x, self.propdown_layer.b)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term - vbias_term   
        return F

    def activation(self, x):
        def act(x):
            return x
        return act

    def sample(self, x):
        def sample(x):
            return x
        return sample

class GaussianUnits():

    def free_energy(self):
        def F(x):
            # gaussian visibles, assumes sigma=1.
            wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
            vbias_term = 0.5*T.sum((x - self.propdown_layer.b)**2, axis=1)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term + vbias_term
        return F

    def activation(self, x):
        def act(x):
            return x
        return act

    def sample(self, x):
        def sample(x):
            return x
        return sample

class Sampling(Layer):

    def __init__(self, sampler=None):
        super(Sampling, self).__init__()
        self.sampler = sampler

    def get_output(self, train=False):
        p = self.get_input(train)
        # TODO add args to the sampler
        return self.sampler(p)


