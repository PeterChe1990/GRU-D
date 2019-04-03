from __future__ import absolute_import, division, print_function

from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine import InputSpec, Layer
from keras.layers.recurrent import _generate_dropout_mask
from keras.layers.recurrent import GRU, GRUCell, RNN
from keras.layers.wrappers import Bidirectional
from keras.utils.generic_utils import has_arg, serialize_keras_object

from .activations import get_activation


__all__ = ['Bidirectional_for_GRUD', 'GRUDCell', 'GRUD']


class GRUDCell(GRUCell):
    """Cell class for the GRU-D layer. An extension of `GRUCell`.
    Notice: Calling with only 1 tensor due to the limitation of Keras.
    Building, computing the shape with the input_shape as a list of length 3.
    # TODO: dynamic imputation
    """

    def __init__(self, units,
                 x_imputation='zero',
                 input_decay='exp_relu', hidden_decay='exp_relu', use_decay_bias=True,
                 feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None,
                 decay_constraint=None,
                 **kwargs):
        super(GRUDCell, self).__init__(units, **kwargs)

        assert 'reset_after' not in kwargs or not kwargs['reset_after'], (
            'Only the default GRU reset gate can be used in GRU-D.'
        )
        assert ('implementation' not in kwargs
                or kwargs['implementation'] == 1), (
                    'Only Implementation-1 (larger number of smaller operations) '
                    'is supported in GRU-D.'
                )

        assert x_imputation in _SUPPORTED_IMPUTATION, (
            'x_imputation {} argument is not supported.'.format(x_imputation)
        )
        self.x_imputation = x_imputation

        self.input_decay = get_activation(input_decay)
        self.hidden_decay = get_activation(hidden_decay)
        self.use_decay_bias = use_decay_bias

        assert (feed_masking or masking_decay is None
                or masking_decay == 'None'), (
                    'Mask needs to be fed into GRU-D to enable the mask_decay.'
                )
        self.feed_masking = feed_masking
        if self.feed_masking:
            self.masking_decay = get_activation(masking_decay)
            self._masking_dropout_mask = None
        else:
            self.masking_decay = None
        
        if (self.input_decay is not None
            or self.hidden_decay is not None
            or self.masking_decay is not None):
            self.decay_initializer = initializers.get(decay_initializer)
            self.decay_regularizer = regularizers.get(decay_regularizer)
            self.decay_constraint = constraints.get(decay_constraint)
        

    def build(self, input_shape):
        """
        Args:
            input_shape: A tuple of 3 shapes (from x, m, s, respectively)
        """
        # Validate the shape of the input first. Borrow the idea from `_Merge`.
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('GRU-D be called on a list of 3 inputs (x, m, s).')
        if input_shape[0] != input_shape[1]:
            raise ValueError('The input x and the masking m should have '
                             'the same input shape, but got '
                             '{} and {}.'.format(input_shape[0], input_shape[1]))
        if input_shape[0][0] != input_shape[2][0]:
            raise ValueError('The input x and the timestamp s should have '
                             'the same batch size, but got '
                             '{} and {}'.format(input_shape[0], input_shape[2]))

        # Borrow the logic from GRUCell for the same part.
        super(GRUDCell, self).build(input_shape[0])

        # Modify the different parts from GRU.
        input_dim = input_shape[0][-1]
        self.state_size = (self.units, input_dim, input_dim)

        # Build the own part of GRU-D.
        if self.input_decay is not None:
            self.input_decay_kernel = self.add_weight(
                shape=(input_dim,),
                name='input_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint
            )
            if self.use_decay_bias:
                self.input_decay_bias = self.add_weight(
                    shape=(input_dim,),
                    name='input_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )
        if self.hidden_decay is not None:
            self.hidden_decay_kernel = self.add_weight(
                shape=(input_dim, self.units),
                name='hidden_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = self.add_weight(
                    shape=(self.units,),
                    name='hidden_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )
        if self.feed_masking:
            self.masking_kernel = self.add_weight(
                shape=(input_dim, self.units * 3),
                name='masking_kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            )
            if self.masking_decay is not None:
                self.masking_decay_kernel = self.add_weight(
                    shape=(input_dim,),
                    name='masking_decay_kernel',
                    initializer=self.decay_initializer,
                    regularizer=self.decay_regularizer,
                    constraint=self.decay_constraint
                )
                if self.use_decay_bias:
                    self.masking_decay_bias = self.add_weight(
                        shape=(input_dim,),
                        name='masking_decay_bias',
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint
                    )
            self.masking_kernel_z = self.masking_kernel[:, :self.units]
            self.masking_kernel_r = self.masking_kernel[:, self.units:self.units * 2]
            self.masking_kernel_h = self.masking_kernel[:, self.units * 2:]

        self.true_input_dim = input_dim
        self.built = True

    def call(self, inputs, states, training=None):
        """We need to reimplmenet `call` entirely rather than reusing that
        from `GRUCell` since there are lots of differences.

        Args:
            inputs: One tensor which is stacked by 3 inputs (x, m, s)
                x and m are of shape (n_batch * input_dim).
                s is of shape (n_batch, 1).
            states: states and other values from the previous step.
                (h_tm1, x_keep_tm1, s_prev_tm1)
        """
        # Get inputs and states
        input_x = inputs[:, :self.true_input_dim]   # inputs x, m, s
        input_m = inputs[:, self.true_input_dim:-1]
        input_s = inputs[:, -1:]
        # Need to add broadcast for time_stamp if using theano backend.
        if K.backend() == 'theano':
            input_s = K.pattern_broadcast(input_s, [False, True])
        h_tm1, x_keep_tm1, s_prev_tm1 = states
        # previous memory ([n_batch * self.units])
        # previous input x ([n_batch * input_dim])
        # and the subtraction term (of delta_t^d in Equation (2))
        # ([n_batch * input_dim])
        input_1m = K.cast_to_floatx(1.) - input_m
        input_d = input_s - s_prev_tm1

        # Get dropout
        if 0. < self.dropout < 1. and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(input_x),
                self.dropout,
                training=training,
                count=3)
        if (0. < self.recurrent_dropout < 1. and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)
        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if self.feed_masking:
            if 0. < self.dropout < 1. and self._masking_dropout_mask is None:
                self._masking_dropout_mask = _generate_dropout_mask(
                    K.ones_like(input_m),
                    self.dropout,
                    training=training,
                    count=3)
            m_dp_mask = self._masking_dropout_mask

        # Compute decay if any
        if self.input_decay is not None:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = K.bias_add(gamma_di, self.input_decay_bias)
            gamma_di = self.input_decay(gamma_di)
        if self.hidden_decay is not None:
            gamma_dh = K.dot(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = K.bias_add(gamma_dh, self.hidden_decay_bias)
            gamma_dh = self.hidden_decay(gamma_dh)
        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = K.bias_add(gamma_dm, self.masking_decay_bias)
            gamma_dm = self.masking_decay(gamma_dm)

        # Get the imputed or decayed input if needed
        # and `x_keep_t` for the next time step

        if self.input_decay is not None:
            x_keep_t = K.switch(input_m, input_x, x_keep_tm1)
            x_t = K.switch(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == 'forward':
            x_t = K.switch(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == 'zero':
            x_t = K.switch(input_m, input_x, K.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == 'raw':
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError('No input decay or invalid x_imputation '
                             '{}.'.format(self.x_imputation))

        # Get decayed hidden if needed
        if self.hidden_decay is not None:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        # Get decayed masking if needed
        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        # Apply the dropout
        if 0. < self.dropout < 1.:
            x_z, x_r, x_h = x_t * dp_mask[0], x_t * dp_mask[1], x_t * dp_mask[2]
            if self.feed_masking:
                m_z, m_r, m_h = (m_t * m_dp_mask[0],
                                 m_t * m_dp_mask[1],
                                 m_t * m_dp_mask[2]
                                )
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z, h_tm1_r = (h_tm1d * rec_dp_mask[0],
                                         h_tm1d * rec_dp_mask[1],
                                        )
        else:
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d

        # Get z_t, r_t, hh_t
        z_t = K.dot(x_z, self.kernel_z) + K.dot(h_tm1_z, self.recurrent_kernel_z)
        r_t = K.dot(x_r, self.kernel_r) + K.dot(h_tm1_r, self.recurrent_kernel_r)
        hh_t = K.dot(x_h, self.kernel_h)
        if self.feed_masking:
            z_t += K.dot(m_z, self.masking_kernel_z)
            r_t += K.dot(m_r, self.masking_kernel_r)
            hh_t += K.dot(m_h, self.masking_kernel_h)
        if self.use_bias:
            z_t = K.bias_add(z_t, self.input_bias_z)
            r_t = K.bias_add(r_t, self.input_bias_r)
            hh_t = K.bias_add(hh_t, self.input_bias_h)
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)
        
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_h = r_t * h_tm1d * rec_dp_mask[2]
        else:
            h_tm1_h = r_t * h_tm1d        
        hh_t = self.activation(hh_t + K.dot(h_tm1_h, self.recurrent_kernel_h))

        # get h_t
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t
        if 0. < self.dropout + self.recurrent_dropout:
            if training is None:
                h_t._uses_learning_phase = True

        # get s_prev_t
        s_prev_t = K.switch(input_m, 
                            K.tile(input_s, [1, self.state_size[-1]]),
                            s_prev_tm1)
        return h_t, [h_t, x_keep_t, s_prev_t]

    def get_config(self):
        # Remember to record all args of the `__init__`
        # which are not covered by `GRUCell`.
        config = {'x_imputation': self.x_imputation,
                  'input_decay': serialize_keras_object(self.input_decay),
                  'hidden_decay': serialize_keras_object(self.hidden_decay),
                  'use_decay_bias': self.use_decay_bias,
                  'feed_masking': self.feed_masking,
                  'masking_decay': serialize_keras_object(self.masking_decay),
                  'decay_initializer': initializers.serialize(self.decay_initializer),
                  'decay_regularizer': regularizers.serialize(self.decay_regularizer),
                  'decay_constraint': constraints.serialize(self.decay_constraint)
                 }
        base_config = super(GRUDCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRUD(GRU):
    """Layer class for the GRU-D. An extension of GRU which utilizes
    missing data for better classification performance.
    Notice: constants is not used in GRUD.
    """

    def __init__(self, units,
                 activation='sigmoid',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 x_imputation='zero',
                 input_decay='exp_relu',
                 hidden_decay='exp_relu',
                 use_decay_bias=True,
                 feed_masking=True,
                 masking_decay=None,
                 decay_initializer='zeros',
                 decay_regularizer=None,
                 decay_constraint=None,
                 **kwargs):

        cell = GRUDCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        x_imputation=x_imputation,
                        input_decay=input_decay,
                        hidden_decay=hidden_decay,
                        use_decay_bias=use_decay_bias,
                        feed_masking=feed_masking,
                        masking_decay=masking_decay,
                        decay_initializer=decay_initializer,
                        decay_regularizer=decay_regularizer,
                        decay_constraint=decay_constraint)
        if 'unroll' in kwargs and kwargs['unroll']:
            raise ValueError('GRU-D does not support unroll.')
        if 'activity_regularizer' in kwargs:
            self.activity_regularizer = regularizers.get(
                kwargs.pop('activity_regularizer'))
        else:
            self.activity_regularizer = None
        # Skip the ` __init__()` of `GRU` and the differences are handled.
        super(GRU, self).__init__(cell, **kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        """Even if `return_state` = True, we do not return x_keep and ss
        (the last 2 states) since they are useless.
        """
        output_shape = super(GRUD, self).compute_output_shape(input_shape)
        if self.return_state:
            return output_shape[:-2]
        return output_shape

    def compute_mask(self, inputs, mask):
        """Even if `return_state` is True, we do not return x_keep and ss
        (the last 2 states) since they are useless.
        """
        output_mask = super(GRUD, self).compute_mask(inputs, mask)
        if self.return_state:
            return output_mask[:-2]
        return output_mask

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states
        # if these are passed in __call__.

        if not isinstance(input_shape, list) or len(input_shape) <= 2:
            raise ValueError('input_shape of GRU-D should be a list of at least 3.')
        input_shape = input_shape[:3]

        batch_size = input_shape[0][0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_shape[0][-1]))
        self.input_spec[1] = InputSpec(shape=(batch_size, None, input_shape[1][-1]))
        self.input_spec[2] = InputSpec(shape=(batch_size, None, 1))

        # allow GRUDCell to build before we set or validate state_spec
        step_input_shape = [(i_s[0],) + i_s[2:] for i_s in input_shape]
        self.cell.build(step_input_shape)

        # set or validate state_spec
        state_size = list(self.cell.state_size)

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs[0])  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        ret = [K.tile(initial_state, [1, dim]) for dim in self.cell.state_size[:-1]]
        # initial_state for s_prev_tm1 should be the same as the first s
        # depending on the direction.
        if self.go_backwards:
            # if go_backwards, we take the last s
            # (we take the largest one in case the padded input can be invalid)
            return ret + [K.tile(K.max(inputs[2], axis=1),
                                 [1, self.cell.state_size[-1]])]
        # otherwise we take the first s.
        return ret + [K.tile(inputs[2][:, 0, :], [1, self.cell.state_size[-1]])]

    def __call__(self, inputs, initial_state=None, **kwargs):
        # We skip `__call__` of `RNN` and `GRU` in this case and directly execute
        # GRUD's great-grandparent's method.
        inputs, initial_state = _standardize_grud_args(inputs, initial_state)

        if initial_state is None:
            return super(RNN, self).__call__(inputs, **kwargs)

        # If `initial_state` is specified and is Keras
        # tensors, then add it to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        kwargs['initial_state'] = initial_state
        additional_inputs += initial_state
        self.state_spec = [InputSpec(shape=K.int_shape(state))
                           for state in initial_state]
        additional_specs += self.state_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = inputs + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        return super(RNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # We need to rewrite this `call` method by combining `RNN`'s and `GRU`'s.
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell._masking_dropout_mask = None

        inputs = inputs[:3]

        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = K.int_shape(inputs[0])[1]

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        def step(inputs, states):
            return self.cell.call(inputs, states, **kwargs)
        # concatenate the inputs and get the mask

        concatenated_inputs = K.concatenate(inputs, axis=-1)
        mask = mask[0]
        last_output, outputs, states = K.rnn(step,
                                             concatenated_inputs,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i, state in enumerate(states):
                updates.append((self.states[i], state))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            states = list(states)[:-2] # remove x_keep and ss
            return [output] + states
        return output

    @property
    def x_imputation(self):
        return self.cell.x_imputation

    @property
    def input_decay(self):
        return self.cell.input_decay

    @property
    def hidden_decay(self):
        return self.cell.hidden_decay

    @property
    def use_decay_bias(self):
        return self.cell.use_decay_bias

    @property
    def feed_masking(self):
        return self.cell.feed_masking

    @property
    def masking_decay(self):
        return self.cell.masking_decay

    @property
    def decay_initializer(self):
        return self.cell.decay_initializer

    @property
    def decay_regularizer(self):
        return self.cell.decay_regularizer

    @property
    def decay_constraint(self):
        return self.cell.decay_constraint

    def get_config(self):
        config = {'x_imputation': self.x_imputation,
                  'input_decay': serialize_keras_object(self.input_decay),
                  'hidden_decay': serialize_keras_object(self.hidden_decay),
                  'use_decay_bias': self.use_decay_bias,
                  'feed_masking': self.feed_masking,
                  'masking_decay': serialize_keras_object(self.masking_decay),
                  'decay_initializer': initializers.get(self.decay_initializer),
                  'decay_regularizer': regularizers.get(self.decay_regularizer),
                  'decay_constraint': constraints.get(self.decay_constraint)}
        base_config = super(GRUD, self).get_config()
        for c in ['implementation', 'reset_after']:
            del base_config[c]
        return dict(list(base_config.items()) + list(config.items()))


class Bidirectional_for_GRUD(Bidirectional):
    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        # We skip the `__call__()` of `Bidirectional`
        # and handle the differences in all cases.

        inputs, initial_state = _standardize_grud_args(
            inputs, initial_state)
        
        if initial_state is None and constants is None:
            return super(Bidirectional, self).__call__(inputs, **kwargs)

        # Applies the same workaround as in `RNN.__call__`
        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            # Check if `initial_state` can be splitted into half
            num_states = len(initial_state)
            if num_states % 2 > 0:
                raise ValueError(
                    'When passing `initial_state` to a Bidirectional RNN, '
                    'the state should be a list containing the states of '
                    'the underlying RNNs. '
                    'Found: ' + str(initial_state))

            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            state_specs = [InputSpec(shape=K.int_shape(state))
                           for state in initial_state]
            self.forward_layer.state_spec = state_specs[:num_states // 2]
            self.backward_layer.state_spec = state_specs[num_states // 2:]
            additional_specs += state_specs
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            constants_spec = [InputSpec(shape=K.int_shape(constant))
                              for constant in constants]
            self.forward_layer.constants_spec = constants_spec
            self.backward_layer.constants_spec = constants_spec
            additional_specs += constants_spec

            self._num_constants = len(constants)
            self.forward_layer._num_constants = self._num_constants
            self.backward_layer._num_constants = self._num_constants

        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state of a Bidirectional'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs

            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(Bidirectional, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        return super(Bidirectional, self).__call__(inputs, **kwargs)

def _standardize_grud_args(inputs, initial_state):
    """Standardize `__call__` to a single list of tensor inputs,
    specifically for GRU-D.

    Args:
        inputs: list/tuple of tensors
        initial_state: tensor or list of tensors or None

    Returns:
        inputs: list of 3 tensors
        initial_state: list of tensors or None
    """
    if not isinstance(inputs, list) or len(inputs) <= 2:
        raise ValueError('inputs to GRU-D should be a list of at least 3 tensors.')
    if initial_state is None:
        if len(inputs) > 3:
            initial_state = inputs[3:]
        inputs = inputs[:3]
    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]
    # end of `to_list_or_none()`
    
    initial_state = to_list_or_none(initial_state)
    return inputs, initial_state


_SUPPORTED_IMPUTATION = ['zero', 'forward', 'raw']

def _get_grud_layers_scope_dict():
    return {
        'Bidirectional_for_GRUD': Bidirectional_for_GRUD,
        'GRUDCell': GRUDCell,
        'GRUD': GRUD,
    }
