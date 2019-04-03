from __future__ import absolute_import, division, print_function

import os

import numpy as np

__all__ = ['DataHandler']


def _filter(ts, max_timestamp=None, max_timesteps=None):
    """
    Args:
        ts: A np.array of n np.array with shape (t_i, d).
        max_timestamp: an Integer > 0 or None.
        max_timesteps: an Integer > 0 or None.

    Returns:
        A np.array of n Integers. Its i-th element (x_i) indicates that
            we will take the first x_i numbers from i-th data sample. 
    """
    if max_timestamp is None:
        ret = np.asarray([len(tt) for tt in ts])
    else:
        ret = np.asarray([np.sum(tt - tt[0] <= max_timestamp) for tt in ts])
    if max_timesteps is not None:
        ret = np.minimum(ret, max_timesteps)
    return ret


def _pad(x, lens):
    """
    Args:
        x: A np.array of n np.array with shape (t_i, d).
        lens: A np.array of n Integers > 0.

    Returns:
        A np.array of shape (n, t, d), where t = min(max_length, max(lens))
    """
    n = len(x)
    t = max(lens)
    d = 1 if x[0].ndim == 1 else x[0].shape[1]
    ret = np.zeros([n, t, d], dtype=float)
    if x[0].ndim == 1:
        for i, xx in enumerate(x):
            ret[i, :lens[i]] = xx[:lens[i], np.newaxis]
    else:
        for i, xx in enumerate(x):
            ret[i, :lens[i]] = xx[:lens[i]]
    return ret

def _rescale(x, mean, std):
    """
    Args:
        x: A np.array of several np.array with shape (t_i, d).
        mean: A np.array of shape (d,).
        std: A np.array of shape (d,).

    Returns:
        Same shape as x with rescaled values.
    """
    return np.asarray([(xx - mean[np.newaxis, :]) / std[np.newaxis, :] for xx in x])


class DataHandler(object):
    """Load `data.npz` and `fold.npz` for model training and testing.
    In `data.npz`:
        Required: `input`, `masking`, `timestamp`, `label_$label_name$`
        Shape: (n_samples,)
    In `fold.npz`:
        Required: `fold_$label_name$`, `mean_$label_name$`, `std_$label_name$`
        Shape: (n_split, 3)
    """
    def __init__(self, data_path, label_name, max_steps=None, max_timestamp=None):
        super(DataHandler, self).__init__()
        self._input_dim = None
        self._output_dim = None
        self._output_activation = None
        self._loss_function = None
        self._folds = None

        self._data_file = os.path.join(data_path, 'data.npz')
        self._fold_file = os.path.join(data_path, 'fold.npz')
        self._load_data(label_name)
        self._max_steps = max_steps
        self._max_timestamp = max_timestamp

    def _load_data(self, label_name):
        if not os.path.exists(self._data_file):
            raise ValueError('Data file does not exist...')
        if not os.path.exists(self._fold_file):
            raise ValueError('Fold file does not exist...')
        # Get input, masking, timestamp, label_$label_name$, fold, mean, std, etc.
        data = np.load(self._data_file)
        fold = np.load(self._fold_file)
        self._data = {}
        for s in ['input', 'masking', 'timestamp']:
            self._data[s] = data[s]
        self._data['label'] = data['label_' + label_name]
        for s in ['fold', 'mean', 'std']:
            self._data[s] = fold[s + '_' + label_name]

        self._input_dim = self._data['input'][0].shape[-1]
        if self._data['label'].ndim == 1:
            self._output_dim = 1
        else:
            self._output_dim = self._data['label'].shape[-1]
        self._output_activation = 'sigmoid'
        self._loss_function = 'binary_crossentropy'
        self._folds = self._data['fold'].shape[0]

    def _get_generator(self, i, i_fold, shuffle, batch_size, return_targets):
        if not return_targets and shuffle:
            raise ValueError('Do not shuffle when targets are not returned.')
        fold = np.copy(self._data['fold'][i_fold][i])
        # The mean / std used in validation/test fold should also be from
        # the training fold.
        mean = self._data['mean'][i_fold][0]
        std = self._data['std'][i_fold][0]
        folds = len(fold)

        def _generator():
            while True:
                if shuffle:
                    np.random.shuffle(fold)
                batch_from = 0
                while batch_from < folds:
                    batch_fold = fold[batch_from:batch_from + batch_size]
                    inputs = [self._data[s][batch_fold] for s
                              in ['input', 'masking', 'timestamp']]
                    inputs[0] = _rescale(inputs[0], mean, std)
                    lens = _filter(inputs[2], self._max_timestamp, self._max_steps)
                    inputs = [_pad(x, lens) for x in inputs]
                    targets = self._data['label'][batch_fold]
                    yield (inputs, targets)
                    batch_from += batch_size
                    print('.', end='')
        # end of `_generator()`

        def _inputs_generator():
            for inputs, _ in _generator():
                yield inputs
        # end of `_inputs_generator()`

        if not return_targets:
            return _inputs_generator()
        return _generator()

    def training_generator(self, i_fold, batch_size):
        return self._get_generator(i=0, i_fold=i_fold, shuffle=True,
                                   batch_size=batch_size, return_targets=True)

    def validation_generator(self, i_fold, batch_size):
        return self._get_generator(i=1, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=True)

    def testing_generator(self, i_fold, batch_size):
        return self._get_generator(i=2, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=True)

    def _steps(self, i, i_fold, batch_size):
        return (self._data['fold'][i_fold][i].size - 1) // batch_size + 1

    def training_steps(self, i_fold, batch_size):
        return self._steps(i=0, i_fold=i_fold, batch_size=batch_size)

    def validation_steps(self, i_fold, batch_size):
        return self._steps(i=1, i_fold=i_fold, batch_size=batch_size)

    def testing_steps(self, i_fold, batch_size):
        return self._steps(i=2, i_fold=i_fold, batch_size=batch_size)

    def training_y(self, i_fold):
        return self._data['label'][self._data['fold'][i_fold][0]]

    def validation_y(self, i_fold):
        return self._data['label'][self._data['fold'][i_fold][1]]

    def testing_y(self, i_fold):
        return self._data['label'][self._data['fold'][i_fold][2]]

    def training_generator_x(self, i_fold, batch_size):
        return self._get_generator(i=0, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    def validation_generator_x(self, i_fold, batch_size):
        return self._get_generator(i=1, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    def testing_generator_x(self, i_fold, batch_size):
        return self._get_generator(i=2, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    @property
    def folds(self):
        return self._folds

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def loss_function(self):
        return self._loss_function
