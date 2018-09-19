from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 moving_stats=False,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.moving_stats = moving_stats
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(
            moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        if self.moving_stats:
            self.moving_mean = self.add_weight(shape=shape,
                                               name='moving_mean',
                                               initializer=self.moving_mean_initializer,
                                               trainable=False)
            self.moving_variance = self.add_weight(shape=shape,
                                                   name='moving_variance',
                                                   initializer=self.moving_variance_initializer,
                                                   trainable=False)
        else:
            self.moving_mean = None
            self.moving_variance = None

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]
        del reduction_axes[0]

        ndim = len(input_shape)
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[1:-1])

        if needs_broadcasting:
            if self.moving_stats:
                broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
                broadcast_moving_var = K.reshape(self.moving_variance, broadcast_shape)
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
        else:
            broadcast_moving_mean = self.moving_mean
            broadcast_moving_var = self.moving_variance
            broadcast_gamma = self.gamma
            broadcast_beta = self.beta

        if self.moving_stats:
            mean = broadcast_moving_mean
            stddev = K.sqrt(broadcast_moving_var) + self.epsilon
        else:
            # Paper version
            #stddev = K.std(inputs, reduction_axes,
            #               keepdims=True) + self.epsilon
            #Pytorch version
            mean = K.mean(inputs, reduction_axes, keepdims=True)
            var = K.var(inputs, reduction_axes, keepdims=True)
            stddev = K.sqrt(var + self.epsilon)

        normed = (inputs - mean) / stddev

        if self.scale:
            normed = normed * broadcast_gamma
        if self.center:
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'moving_stats': self.moving_stats,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'InstanceNormalization': InstanceNormalization})
