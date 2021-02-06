

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Multiply

tfd = tfp.distributions


class VariationalWeibullLayer(tf.keras.Model):
    def call(self, input_tensor, training=False):
        
        # X ~ Chi-Squared(0.1) -> c*X ~ Gamma(0.1/2, 2c)
        # U ~ Uniform(0, 1) -> n*U ~ Uninform(0, n)
        # Let  J = sum(cX_i ~ Gamma(1, 2c)), i =1...U then
        # J ~ Gamma((0.1/2)*U, 2c)
        # Finally 1/J ~ Inv-Gamma((0.1/2)*U, 2c)

        lambda_pipe = self.x1(input_tensor)
        lambda_pipe = self.lamb(lambda_pipe)

        k_pipe = self.x2(input_tensor)
        k_pipe = self.k(k_pipe) + tf.constant(1.)

        return self._result(lambda_pipe, k_pipe)


class WeibullLayer(tf.keras.Model):
    def __init__(self, return_params=True, regression='mode', name=''):
        super().__init__(name=name)
        self.return_params = return_params
        if self.return_params:
            self.params = Concatenate(name='Weibullparams')
        if regression == 'mode':
            self.fun = self.mode
        elif regression == 'mean':
            self.fun = self.mean
        elif regression == 'median':
            self.fun = self.median

    def mean(self, lambda_pipe, k_pipe):
        inner_gamma = Lambda(
            lambda x: tf.math.exp(tf.math.lgamma(1+(1/x))))(k_pipe)
        return Multiply(name='RUL')([lambda_pipe, inner_gamma])

    def median(self, lambda_pipe, k_pipe):
        return lambda_pipe*(tf.math.pow(tf.math.log(2.), tf.math.reciprocal(k_pipe)))

    def mode(self, lambda_pipe, k_pipe):
        def replacenan(t):
            return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

        def mode(k):
            mask = K.cast(K.greater(k, 1), tf.float32)
            aa = K.pow((k-1)/(k), 1/(k))
            b = replacenan(tf.math.multiply(mask,  aa))
            return b

        i = Lambda(mode)(k_pipe)

        return Multiply(name='RUL')([lambda_pipe, i])

    def _result(self, lambda_, k):
        RUL = self.fun(lambda_, k)
        if self.return_params:
            return [RUL, self.params([lambda_, k])]
        else:
            return RUL


class WeibullParameters(WeibullLayer):
    def __init__(self, hidden, regression='mode', return_params=True):
        super(WeibullParameters, self).__init__(
            return_params=True, regression=regression, name='')
        self.x1 = Dense(hidden,
                        activation='relu')
        self.lamb = Dense(1,
                          activation=tf.math.exp,
                          name='w_lambda')

        self.x2 = Dense(hidden,
                        activation='relu')
        self.k = Dense(1,
                       activation='softplus',
                       name='w_k')

        self.return_params = return_params

    def call(self, input_tensor, training=False):
        lambda_pipe = self.x1(input_tensor)
        lambda_pipe = self.lamb(lambda_pipe)

        k_pipe = self.x2(input_tensor)
        k_pipe = self.k(k_pipe) + tf.constant(1.)

        return self._result(lambda_pipe, k_pipe)
