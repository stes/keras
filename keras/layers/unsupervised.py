from ..layers.core import Layer, Dense
from .. import activations, initializations, regularizers, constraints
from ..layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from ..layers import containers
from ..layers.tied import TiedDense, TiedConvolution2D, transpose

import keras.backend as K

border_transp = {'same': 'same', 'valid': 'full', 'full': 'valid'}

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

    def activation(self):
        def act(x):
            return x
        return act

    def sample(self):
        def sample(x):
            return x
        return sample

class BernoulliUnits(RBMUnits):

    def free_energy(self):
        def F(x):
            wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
            vbias_term = T.dot(x, self.propdown_layer.b)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term - vbias_term   
        return F

    def activation(self):
        return activations.get('sigmoid')

    def sample(self):
        return samplers.get('bernoulli')

class GaussianUnits(RBMUnits):

    def free_energy(self):
        def F(x):
            # gaussian visibles, assumes sigma=1.
            wx_b = T.dot(x, self.propup_layer.W) + self.propup_layer.b
            vbias_term = 0.5*T.sum((x - self.propdown_layer.b)**2, axis=1)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term + vbias_term
        return F

    def activation(self):
        def act(x):
            return x
        return act

    def sample(self, mode='sample'):
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

