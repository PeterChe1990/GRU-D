from __future__ import absolute_import, division, print_function

from keras import backend as K
from keras.layers import Activation, Dense, Dropout, Input, Masking
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.models import load_model, Model
from keras.regularizers import l2
from keras.utils.generic_utils import custom_object_scope

from nn_utils.grud_layers import Bidirectional_for_GRUD, GRUD
from nn_utils.layers import ExternalMasking


__all__ = ['create_grud_model', 'load_grud_model']


def create_grud_model(input_dim, recurrent_dim, hidden_dim,
                      output_dim, output_activation,
                      predefined_model=None,
                      use_bidirectional_rnn=False, use_batchnorm=False, **kwargs):

    if (predefined_model is not None
            and predefined_model in _PREDEFINED_MODEL_LIST):
        for c, v in _PREDEFINED_MODEL_LIST[predefined_model].items():
            kwargs[c] = v
    # Input
    input_x = Input(shape=(None, input_dim))
    input_m = Input(shape=(None, input_dim))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]
    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)
    # GRU layers
    grud_layer = GRUD(units=recurrent_dim[0],
                      return_sequences=len(recurrent_dim) > 1,
                      activation='sigmoid',
                      dropout=0.3,
                      recurrent_dropout=0.3,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    x = grud_layer([input_x, input_m, input_s])
    for i, rd in enumerate(recurrent_dim[1:]):
        gru_layer = GRU(units=rd,
                        return_sequences=i < len(recurrent_dim) - 2,
                        dropout=0.3,
                        recurrent_dropout=0.3,
                       )
        if use_bidirectional_rnn:
            gru_layer = Bidirectional(gru_layer)
        x = gru_layer(x)
    # MLP layers
    x = Dropout(.3)(x)
    for hd in hidden_dim:        
        x = Dense(units=hd,
                  kernel_regularizer=l2(1e-4))(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Dense(output_dim, activation=output_activation)(x)
    output_list = [x]

    model = Model(inputs=input_list, outputs=output_list)
    return model


def load_grud_model(file_name):
    from nn_utils import _get_scope_dict
    with custom_object_scope(_get_scope_dict()):
        model = load_model(file_name)
    return model


_PREDEFINED_MODEL_LIST = {
    'GRUD': {
        'x_imputation': 'zero',
        'input_decay': 'exp_relu',
        'hidden_decay': 'exp_relu',
        'feed_masking': True,
    },
    'GRUmean': {
        'x_imputation': 'zero',
        'input_decay': None,
        'hidden_decay': None,
        'feed_masking': False,
    },
    'GRUforward': {
        'x_imputation': 'forward',
        'input_decay': None,
        'hidden_decay': None,
        'feed_masking': False,
    },
    'GRUsimple': {
        'x_imputation': 'zero',
        'input_decay': None,
        'hidden_decay': None,
        'feed_masking': True,
    },
}
