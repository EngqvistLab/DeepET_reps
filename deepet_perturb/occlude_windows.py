import argparse
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import pandas as pd

from deepet_perturb.general import data, models, util


def main():
    args = get_args()
    logger = util.get_logger()

    sequences_df = data.fasta_to_seq_df(args.input, ['seq_id', 'seq'])
    sequences_df['seq_id'] = sequences_df['seq_id'].map(
        lambda seq_id: seq_id.split(';')[0]
    )
    sequences = sequences_df['seq'].values
    sequence_ids = sequences_df['seq_id'].values

    wt_preds, changes, z_scores = models.occlude(
        sequences,
        window=args.window,
        model_type='topt',
        model_filename='data/models/bestmodel_c3f1d2.h5',
        layer='flatten_1'
    )

    with h5py.File(args.output, 'w') as fres:
        for i in range(sequences.shape[0]):
            if z_scores[i] is not None:
                prot_group = fres.create_group(sequence_ids[i])
                prot_group.create_dataset('z_scores', data=np.array(z_scores[i], dtype=np.float32))
                prot_group.create_dataset('changes', data=np.array(changes[i], dtype=np.float32))
                prot_group.create_dataset('wt_preds', data=np.array(wt_preds[i], dtype=np.float32))
            else:
                logger.warning(sequence_ids[i] + ' was too long. Skipped.')
    logger.info('Done. Saved results to H5 store and MSA fasta')


def get_args():
    parser = argparse.ArgumentParser(
        description='Occldue all sequences in input FASA with sliding windows'
    )
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Output H5 store filename')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Input FASTA file name')
    parser.add_argument('-w',
                        '--window',
                        required=False,
                        type=int,
                        default=5,
                        help='Window length (default: %(default)s)')
    return parser.parse_args()


if __name__ == '__main__':
    main()
