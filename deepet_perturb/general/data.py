"""
Data handling routines
"""
import sys

from Bio import SeqIO
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def fasta_to_seq_df(filename: str, column_names: list) -> pd.DataFrame:
    return pd.DataFrame([[entry.id, str(entry.seq)]
                          for entry in list(SeqIO.parse(str(filename), 'fasta'))],
                        columns=column_names)


def load_enzyme_data(filename: str) -> pd.DataFrame:
    """
    Convenience func: return DataFrame with columns
    [seq, uniprot_ac, ogt, topt, seq_len]
    """
    enz_data = fasta_to_seq_df(str(filename), column_names=['enz_id', 'seq'])
    enz_data = enz_data.assign(
        uniprot_ac=enz_data['enz_id'].map(lambda eid: eid.split(';')[0]),
        ogt=enz_data['enz_id'].map(lambda eid: np.float16(eid.split(';')[1].split('=')[1])),
        topt=enz_data['enz_id'].map(lambda eid: np.float16(eid.split(';')[2].split('=')[1])),
        seq_len=enz_data['seq'].str.len()
    ).drop(columns='enz_id')
    return enz_data


def get_dssp_annotation(pdb_file: str, seq_len: int) -> str:
    """
    Return amino acid sequence annotation with secondary structure, using DSSP.
    Use only first model only in the PDB file.

    Coil is denoted as '-' and missing residues are denoted as '_'
    """
    try:
        dssp_dict = dssp_dict_from_pdb_file(str(pdb_file))[0]
        annotation = ['_'] * seq_len
        for res_id, res_features in dssp_dict.items():
            res_idx = res_id[1][1]
            struct = res_features[1]
            annotation[res_idx - 1] = struct
        return ''.join(annotation)

    except IndexError as e:
        print('Residue indexing error for ' + str(pdb_file), file=sys.stderr)
        return None

    except Exception as e:
        print('Error running DSSP on ' + str(pdb_file), file=sys.stderr)
        return None


def upsample(vector: np.ndarray, max_len: int) -> np.array:
    """
    Return upsampled vector to max_len (using linear interpolation)
    """
    vector_interpolation = interp1d(np.arange(0, vector.shape[0]),
                                    vector, kind='linear')
    new_range = np.linspace(0, vector.shape[0] - 1, num=max_len)
    upsampled_vector = vector_interpolation(new_range)
    return upsampled_vector
