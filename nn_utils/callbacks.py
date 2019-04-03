"""Customized callbacks for Keras models
"""
from __future__ import absolute_import, division, print_function

from datetime import datetime
import os

from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np


__all__ = ['ModelCheckpointwithBestWeights', 'TensorBoardwithValidationData']


class ModelCheckpointwithBestWeights(ModelCheckpoint):
    """Model checkpoint which can restore the best weights at the end of the training.
    """
    def __init__(self, file_dir='.', file_name=r'weights-{epoch}.h5', verbose=0, **kwargs):
        """Note: Interface of keras.callbacks.ModelCheckpoint:
            __init__(self, filepath, monitor='val_loss', verbose=0,
                     save_best_only=False, save_weights_only=False,
                     mode='auto', period=1)
        """
        kwargs['save_best_only'] = True
        kwargs['save_weights_only'] = True
        self.verbose_this = verbose
        kwargs['verbose'] = verbose == 1
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filepath = os.path.join(
            file_dir,
            file_name.format(timestamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
                             epoch='{epoch}')
        )
        super(ModelCheckpointwithBestWeights, self).__init__(filepath=filepath, **kwargs)
        self.prev_best = self.best
        self.best_epoch = -1
        self.best_filepath = ""
        self.temp_filepaths = []

    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, if the current epoch provides the best model,
        save the weight files.
        """
        super(ModelCheckpointwithBestWeights, self).on_epoch_end(epoch=epoch, logs=logs)
        self.temp_filepaths.append(self.filepath.format(epoch=epoch+1, **logs))
        if self.best != self.prev_best:
            self.best_epoch = epoch
            self.prev_best = self.best
            self.best_filepath = self.temp_filepaths[-1]

    def on_train_end(self, logs=None):
        """ At the end of training, try to restore the best weights and remove
        other weight files.
        """
        if self.best_epoch >= 0:
            self.model.load_weights(self.best_filepath)
            self.temp_filepaths.remove(self.best_filepath)
        for tfp in self.temp_filepaths:
            if os.path.exists(tfp):
                os.remove(tfp)

class TensorBoardwithValidationData(TensorBoard):
    def __init__(self, validation_data, learning_phase=None, **kwargs):
        super(TensorBoardwithValidationData, self).__init__(**kwargs)
        # Borrow from validation data preparation in fit_generator
        # https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py#L105
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        else:
            raise ValueError('`validation_data` can only be `(val_x, val_y)`. Found: ',
                             str(validation_data))
        val_data = []
        if isinstance(val_x, list):
            val_data += val_x
        else:
            val_data += [val_x]
        if isinstance(val_y, list):
            val_data += val_y
        else:
            val_data += [val_y]
        val_data += [
            np.ones((val_data[0].shape[0],), dtype=K.floatx()),
            0.,
        ]
        self.validation_data = val_data

def _get_callbacks_scope_dict():
    return {
        'ModelCheckpointwithBestWeights': ModelCheckpointwithBestWeights,
        'TensorBoardwithValidationData': TensorBoardwithValidationData,
    }