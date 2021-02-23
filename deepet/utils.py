import numpy as np

def to_binary(seq):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(20, dtype = np.float32)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((len(seq),20), dtype = np.float32)
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(20, dtype = np.float32))
        seq_coding[i,:] = code
    return seq_coding


def zero_padding(inp, length=2000, start=False):
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
