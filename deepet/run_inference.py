
import numpy as np
import pandas as pd
from os.path import join

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from Bio import SeqIO
from Bio import AlignIO

from deepet import my_callbacks
from pkg_resources import resource_stream, resource_filename, resource_exists

ALANINE = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # sets global vareable ALANINE as a one hot reprecentation of alanine
NOTHING = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # sets global vareable NOTHING as a one hot reprecentation that's empty


class DeepETInference(object):
    '''
    A class for getting DeepET embeddings.
    '''
    def __init__(self, model, layer):

        # load the model
        assert model in ['ogt', 'topt']
        self.model_type = model

        if model == 'ogt':
            model_path = resource_filename(__name__, "best_ogt_model")
            self.model = load_model(join(model_path, 'bestmodel.h5'), custom_objects={'coef_det_k':my_callbacks.coef_det_k})

        elif model == 'topt':
            model_path = resource_filename(__name__, "best_topt_model")
            self.model = load_model(join(model_path, 'bestmodel.h5'), custom_objects={'coef_det_k':my_callbacks.coef_det_k})

        else:
            raise ValueError

        # make a smaller one to get the embeddings
        assert layer in ['flatten_1', 'dense_1', 'dense_2']
        self.layer = layer
        self.build_bottleneck_model(model=self.model, layer_name=self.layer)


    def run_inference(self, filepath):
        '''
        Accept fasta file path and make predictions for each of these.
        Return data frame.
        '''
        # read sequences into a Pandas series with sequences and identifiers
        seqs = self._series_from_seqio(filepath, 'fasta')
        seqs = seqs.str.rstrip('*')

        # one-hot encoding of sequences
        encoded_seqs = self.seq_to_encoding(seqs)

        # get sequence predictions
        predictions = self.model.predict(encoded_seqs)

        return pd.DataFrame(predictions)


    def get_embeddings(self, filepath):
        '''
        Accept fasta file path and obtain embeddings for each of these.
        Return data frame.
        '''
        # read sequences into a Pandas series with sequences and identifiers
        seqs = self._series_from_seqio(filepath, 'fasta')
        seqs = seqs.str.rstrip('*')

        # one-hot encoding of sequences
        encoded_seqs = self.seq_to_encoding(seqs)

        # get sequence embeddings
        embeddings = self.bottleneck_model.predict(encoded_seqs)

        return pd.DataFrame(embeddings)


    def occlusion_1d(self, seq, window, padding=2000):
        '''
        Occlude a single sequence using a sliding window.
        Return the original temp, temperature change and a
        z-score for each position indicating sensititivity.
        '''
        seq = self._to_binary(seq.rstrip('*'))
        size_seq = len(seq)

        # first predict the unchanged sequence
        original_seq = self._zero_padding(seq, padding)
        wt_pred = self.model.predict( original_seq.reshape([1, original_seq.shape[0], original_seq.shape[1]]) )[0]

        seqs = []
        for i in range(0, size_seq - window):

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

        # compute the average prediction for each position
        window_average = np.convolve(predictions.reshape(-1), np.ones(window)/window, mode='valid')

        # what's the change in temperature?
        change = (wt_pred - window_average)

        # compute a z-score
        z_score = (change-np.mean(change))/np.std(change)

        return wt_pred, change, z_score



    def _series_from_seqio(self, fn, format, **kwargs):
        '''
        '''
        if format in SeqIO._FormatToIterator.keys():
            reader = SeqIO.parse
        elif format in AlignIO._FormatToIterator.keys():
            reader = AlignIO.read
        else:
            raise ValueError("format {} not recognized by either SeqIO or AlignIO".format(format))

        if isinstance(fn, str) and 'gz' in fn:
            with gzip.open(fn, "rt") as fh:
                seqs = reader(fh, format, *kwargs)
        else:
            seqs = reader(fn, format, *kwargs)

        seqs = [(r.description, str(r.seq).upper()) for r in seqs]
        seqs = list(zip(*seqs))
        seqs = pd.Series(seqs[1], index=seqs[0], name="seq")

        return seqs


    def build_bottleneck_model(self, model, layer_name):
        '''
        Take full model and truncate at flatten or dense layers
        '''
        for layer in model.layers:
            if layer.name == layer_name:
                output = layer.output
        self.bottleneck_model = Model(model.input, output)



    def print_model(self):
        '''
        Print out the layers of a model
        '''
        print('Layers in base model:')
        for layer in self.bottleneck_model.layers:
            print(layer.name)
        print('')


    def _to_binary(self, seq):
        '''
        Eoncode non-standard amino acids like X as all zeros
        output a array with size of L*20
        '''
        seq = seq.upper()
        aas = 'ACDEFGHIKLMNPQRSTVWY'
        pos = dict()
        for i in range(len(aas)): pos[aas[i]] = i

        binary_code = dict()
        for aa in aas:
            code = np.zeros(20)
            code[pos[aa]] = 1
            binary_code[aa] = code

        seq_coding = np.zeros((len(seq),20))
        for i,aa in enumerate(seq):
            code = binary_code.get(aa,np.zeros(20))
            seq_coding[i,:] = code
        return seq_coding


    def _zero_padding(self, inp, length, start=False):
        '''
        zero pad input one hot matrix to desired length
        start .. boolean if pad start of sequence (True) or end (False)
        '''
        assert len(inp) <= length
        out = np.zeros((length,inp.shape[1]))
        if start:
            out[-inp.shape[0]:] = inp
        else:
            out[0:inp.shape[0]] = inp
        return out


    def seq_to_encoding(self, seqs, length_cutoff = 2000):
        '''
        Process amino acid sequence, preparing it as input to the network.
        '''
        encodings = []
        for seq in seqs:
            encodings.append(self._zero_padding(self._to_binary(seq), length_cutoff))
        return np.array(encodings)
