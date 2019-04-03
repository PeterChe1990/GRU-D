from __future__ import absolute_import, division, print_function

import keras
from keras import backend as K
from keras.layers import Masking


__all__ = ['ExternalMasking']


class ExternalMasking(Masking):
    """An extension of `Masking` layer.
    Use the second input to determine the masking of the first input.
    """
    def compute_mask(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Inputs to ExternalMasking should be a list of 2 tensors.')
        return super(ExternalMasking, self).compute_mask(inputs[-1])

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Inputs to ExternalMasking should be a list of 2 tensors.')
        boolean_mask = K.any(K.not_equal(inputs[-1], self.mask_value),
                             axis=-1, keepdims=True)
        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('input_shape to ExternalMasking should be a list of 2 tensors.')
        if input_shape[0][:2] != input_shape[1][:2]:
            raise ValueError('The first two dimensions of the two inputs should be the '
                             'same, but got {} and {} from them.'.format(
                                 input_shape[0][:2], input_shape[1][:2])
                            )
        return input_shape[0]


def _get_layers_scope_dict():
    return {
        'ExternalMasking': ExternalMasking,
    }
