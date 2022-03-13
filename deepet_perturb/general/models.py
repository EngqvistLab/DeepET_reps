"""
Routines and classes to work with models
"""
import os

import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from deepet.run_inference import DeepETInference, NOTHING
from deepet import my_callbacks


def occlude(seq_list, window, model_type, model_filename, layer):
    '''
    Cary out occlusion on sequences in the list.
    The sequences are padded with repeats of the first and last amino acid,
    to composenate for the fact that sliding windows would only cover the edges
    only once and would result in fewer values than the sequence length.

    The edge-effect compensation means sequences that have
    length + window size > model input size will be skipped
    and None appended to the result lists.

    Adapted from Martin's deepet version.
    '''
    inf_obj = AdaptedDeepETInference(model_filename, model_type, layer)
    input_size = inf_obj.model.get_config()['layers'][0]['config']['batch_input_shape'][1]
    wt_preds = []
    changes = []
    z_scores = []
    for seq in tqdm(seq_list, desc='Sequences occluded'):
        if len(seq) + window <= input_size:
            n_res_left = window - 1 // 2
            n_res_rigt = window - 1 - n_res_left
            seq = seq[0] * n_res_left + seq + seq[-1] * n_res_rigt
            wt_pred, change, z_score = inf_obj.occlusion_1d(seq, window=window)
            wt_preds.append(wt_pred)
            changes.append(change)
            z_scores.append(z_score)
        else:
            wt_preds.append(None)
            changes.append(None)
            z_scores.append(None)
    return wt_preds, changes, z_scores


class AdaptedDeepETInference(DeepETInference):
    """
    Minor adaptations to help run with different project structures.
    """
    def __init__(self, model_filename, model_type, layer, rnd_seed=42):
        assert model_type in ['ogt', 'topt']
        assert layer in ['flatten_1', 'dense_1', 'dense_2']

        np.random.seed(rnd_seed)
        tf.compat.v1.set_random_seed(rnd_seed)
        os.environ['PYTHONHASHSEED'] = '0'

        self.model_type = model_type
        self.model = load_model(model_filename,
                                custom_objects={'coef_det_k': my_callbacks.coef_det_k})

        # Make a smaller model to get the embeddings
        self.layer = layer
        self.build_bottleneck_model(model=self.model, layer_name=self.layer)


    def occlusion_1d(self, seq, window, padding=2000):
        '''
        Occlude a single sequence using a sliding window.
        Return the original predicted temp, temperature
        change and a z-score for each position
        indicating sensititivity.
        '''
        size_seq = len(seq)
        seq = self._to_binary(seq.rstrip('*'))

        # first predict the unchanged sequence
        original_seq = self._zero_padding(seq, padding)
        in_shape = [1, original_seq.shape[0], original_seq.shape[1]]
        wt_pred = self.model.predict(original_seq.reshape(in_shape))[0]

        seqs = []
        for i in range(0, size_seq - window + 1):

            # create empty array same shape as the one-hot sequence
            tmp = np.zeros((seq.shape[0], seq.shape[1]))

            # one-dimensional vector to index which amino acids to replace
            logic = np.zeros(seq.shape[0], dtype = bool)
            logic[i:i+window] = True

            # do the replacement
            tmp[logic,:] = NOTHING
            tmp[~logic,:] = seq[~logic,:]

            # pad sequence to full length
            tmp = self._zero_padding(tmp, padding)
            seqs.append(tmp)

        seqs = np.array(seqs)

        # predict for the occluded sequences
        predictions = self.model.predict(seqs)

        # compute moving average across the vector of perturbation predictions
        window_moving_average = uniform_filter1d(predictions.reshape(-1),
                                                 size=window,
                                                 mode='nearest')

        # compute the change in temperature, as a fraction of the WT prediction
        change = (wt_pred - window_moving_average) / wt_pred

        # compute a z-score
        z_score = (change-np.mean(change))/np.std(change)

        return wt_pred, change, z_score
