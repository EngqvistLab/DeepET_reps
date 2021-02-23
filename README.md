# Description
DeepET model from Li et al.

## Requirements
Assuming that you use Miniconda (https://docs.conda.io/en/latest/miniconda.html) or Anaconda (https://www.anaconda.com/) the required packages can be easily installed from the yml file `packages.yml`. In a terminal execute:
```bash
conda env create -f packages.yml
conda activate deepet
```

## Installation
Download repository and unzip (alternatively fork or clone), cd to the project base folder and execute the command below:

```bash
pip3 install -e .
```

If using an anaconda environment you may have to first locate the miniconda pip using whereis.
```bash
whereis pip
```

Locate the appropriate file path (the one that has anaconda and the correct environment in the filepath) and run the modified command. For example:

```bash
/home/username/miniconda3/envs/py37/bin/pip install -e .
```

The library should now be available for loading in all your python scripts.


## Usage

```python
from deepet.run_inference import DeepETInference
inf_obj = DeepETInference(model='', layer='')
df = inf_obj.run_inference(filepath='my_sequences.fasta')
df.to_csv('my_sequences_embeddings.tsv', sep='\t')
```

Valid values for model are: 'ogt' and 'topt'

Valid values for layer are: 'flatten_1', 'dense_1' and 'dense_2'
