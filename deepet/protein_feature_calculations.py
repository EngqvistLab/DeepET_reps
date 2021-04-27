import math
import pandas as pd
from itertools import combinations_with_replacement
import os
from os.path import join, abspath, basename, isfile, exists
from collections import defaultdict
import numpy as np
import itertools
import subprocess
import re

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser, is_aa, Select, PDBIO
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.Align.Applications import MuscleCommandline
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import io

import warnings

warnings.filterwarnings("ignore", category=PDBConstructionWarning)



class CleanSelect(Select):
    '''
    Class for selecting the parts of the structure that I want to save.
    Subclassing the Select class from Bio.PDB,
    as described about midway down at this page:
    https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
    '''
    def accept_model(self, model):
        '''
        Keep only first model (important for NMR structures)
        '''
        if model.get_id() == 0:
            return 1
        return 0

    def accept_chain(self, chain):
        '''
        Accept all chains
        '''
        return 1

    def accept_residue(self, residue):
        '''
        Keep only amino acid residues, this cleans away DNA and RNA
        Remove all HETATOMS
        '''
        # the 22 natural aa
        aa = ['ALA', 'ARG',
             'ASN', 'ASP',
             'CYS', 'GLN',
             'GLU', 'GLY',
             'HIS', 'ILE',
             'LEU', 'LYS',
             'MET', 'PHE',
             'PRO', 'PYL',
             'SEC', 'SER',
             'THR', 'TRP',
             'TYR', 'VAL']

        # keep amino acids
        if residue.get_resname() in aa:

            # skip HETATOMS
            hetatm_flag, resseq, icode = residue.get_id()
            if hetatm_flag == " ":
                return 1
        return 0

    def accept_atom(self, atom):
        '''
        Remove hydrogens
        Keep only the first atom position for disordered atoms.
        '''
        # first check whether it is a hydrogen or not
        hydrogen = re.compile("[123 ]*H.*")
        name = atom.get_id()
        if not hydrogen.match(name):

            # now skip all alternate locations (keep only first instance)
            if (not atom.is_disordered()) or atom.get_altloc() == 'A':
                atom.set_altloc(' ')  # Eliminate alt location ID before output.
                return 1
        return 0


class TriominoesParser(object):
    '''
    For running tiominoes and parsing the output
    to get coordinates and interactions.
    '''
    def __init__(self, filepath, metadata='standard'):
        assert filepath.lower().endswith('.pdb')
        self.filepath = filepath

        if metadata == 'standard':
            self.metadata_path = './external_programs/Triominoes/metadata.txt'
        else:
            raise ValueError

        # run progarm and parse the output
        self.__run_triominoes()
        self.__parse_output()

    def __run_triominoes(self):
        '''
        Run the triominoes program on a pdb file
        and obtain the output.
        '''
        result = subprocess.run(['./external_programs/Triominoes/triominoes', '-m', self.metadata_path, self.filepath], stdout=subprocess.PIPE)
        self.data = result.stdout.decode('utf8').split('\n')

    def __parse_output(self):
        '''
        Parse each triangle to get the participating atoms,
        the coordinates, and the normals.
        '''
        self.all_coords = []
        self.all_normals = []
        self.all_interactions_atoms = []
        self.all_interactions_numbers = []

        for line in self.data:
            if line == '':
                continue
            i, num1, a1, num2, a2, num3, a3, point, normal = line.split()

            self.all_coords.append(list(map(float, point.replace('point(', '').replace(')', '').split(','))))
            self.all_normals.append(list(map(float, normal.replace('normal(', '').replace(')', '').split(','))))
            self.all_interactions_atoms.append([a1, a2, a3])
            self.all_interactions_numbers.append([int(num1), int(num2), int(num3)])

        self.all_coords = np.array(self.all_coords, dtype='float32')
        self.all_normals = np.array(self.all_normals, dtype='float32')


class PhiPsiParser(object):
    '''
    For running phipsi and parsing the output.
    '''
    def __init__(self):
        raise NotImplementedError


class PDBTools(object):
    '''
    Base class containing methods used by the other classses.
    '''
    def __init__(self):
        self.threetoone = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
                      'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
                      'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
                      'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

        self.onetothree = dict([(self.threetoone[x], x) for x in self.threetoone.keys()])


    def calc_dist_2d(self, c1, c2):
        '''
        Calculates the distance between two coordinates.
        Coordinates need shape as (x,y)
        '''
        dist = math.sqrt((c2[0] - c1[0]) ** 2 +
                         (c2[1] - c1[1]) ** 2)

        return dist


    def calc_dist_3d(self, c1, c2):
        '''
        Calculates the distance between two coordinates.
        Coordinates need shape as (x,y,z).
        '''
        dist = math.sqrt((c2[0] - c1[0]) ** 2 +
                         (c2[1] - c1[1]) ** 2 +
                         (c2[2] - c1[2]) ** 2)
        return dist


    def calc_dist_vectorized_all_atom_pairs(self, coords1, coords2):
       '''
       Vectorized version of calc_dist_3d, calculate all distances for set of coordinates.
       '''
       # np.nansum ??
       return np.sqrt(np.sum(np.power(coords1[np.newaxis, :] - coords2[:, np.newaxis], 2), axis=2))


    def calc_min_dist_vectorized_all_atom_pairs(self, coords1, coords2):
       '''
       Get the minimum distance between a set of coordinates.
       '''
       return np.min(self.calc_dist_vectorized_all_atom_pairs(coords1, coords2))



class PDBReader(PDBTools):
    """
    Functions for parsing PDB files, obtaining atom coordinates and
    residue names.

    Parameters
    ----------

    filepath : string
        Path where the PDB file is located.

    """
    def __init__(self, filepath, clean_filepath):
        super().__init__()

        assert filepath.lower().endswith('.pdb')
        self.filepath = filepath

        assert clean_filepath.lower().endswith('.pdb')
        self.clean_filepath = clean_filepath

        # get the identifier
        pdb_id = basename(self.filepath.upper()).replace('.PDB', '')

        # generate a cleaned pdb file, if it does not exist
        if not exists(self.clean_filepath):
            try:
                # parse
                structure = PDBParser().get_structure(id='infile', file=self.filepath)

                # save the structure, but without the things I don't want
                io = PDBIO()
                io.set_structure(structure)
                io.save(self.clean_filepath, CleanSelect())
            except:
                print('{} parsing failed when cleaning pdb'.format(pdb_id))

        # setup data variables
        self.filename = pdb_id
        self.all_atoms_names = {} # dictionary with chain_id keys holding list with all atom names
        self.all_atoms_coords = {} # dictionary with chain_id keys holding list with all atom coordinates
        self.all_atomic_groups = {} # dictionary with chain_id keys holding list with all atomic groups (13 possible)

        self.res_atoms = {} # dictionary with chain_id keys holding list of lists with the atom names for each amino acid
        self.res_atoms_coords = {} # dictionary with chain_id keys holding list of lists with the atom coordinates for each amino acid
        self.res = {} # dictionary with chain_id as keys holding list of the one-letter amino acids as they occur in structure
        self.res_seqnum = {} # dictionary with chain_id as keys holding list of amino acid numbers (position in protein seq)

        self.seqres = {} # dictionary with chain_id keys holding a list with all residue names (of original protein, some may be missing in structure)
        self.seqres_seqnum = {} # dictionary with chain_id keys holding a list amino acid numbers (position in original protein, some may be missing in structure)

        self.structure_gaps = {} # dictionary with chain_id keys holding a list of postions that are missing

        self.chain_length = {} # dictionary with chain_id as keys holding the chain length as an integer
        self.a_coords = {} # dictionary with chain_id keys holding list alpha carbon coordinates for each amino acid
        self.b_coords = {} # dictionary with chain_id keys holding list beta carbon coordinates for each amino acid

        self.type = 'N/A'
        self.ec_number = 'N/A'
        self.length = 0

        # a list of all amino acid atom ids
        self.atom_ids = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2',
                            'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1',
                            'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N',
                            'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1',
                            'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
                            'OE2', 'OG', 'OG1', 'OH', 'SD', 'SG']

        # a dictionary to obtain atomic group given amino acid and atom id
        self.atomic_groups = self.__parse_atomic_groups()

        # get the "true" protein sequence (some of which may be missing in structure)
        self.__get_seqres()

        # parse the pdb
        self.structure = PDBParser().get_structure(id=pdb_id.upper(), file=self.clean_filepath)
        self.__parse()

        # get structure gaps (find out for which positions amino acids are missing)
        self.__align_seqs()


    def __parse_atomic_groups(self):
        '''
        Obtain a dictionary that maps amino acid and atom id to
        a set of 13 atomic groups. As defined in Tsai, J., Taylor, R.,
        Chothia, C., & Gerstein, M. (1999). The packing density in
        proteins: standard radii and volumes. Journal of molecular biology,
        290(1), 253-266.
        '''
        filepath = join('.', 'data', 'atoms', 'atomic-groups.txt')
        atomic_groups = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('atomic_group'):
                    continue

                aa, atom_id, ag = line.split()

                if atomic_groups.get(aa) is None:
                    atomic_groups[aa] = {}

                atomic_groups[aa][atom_id] = ag

        return atomic_groups


    def __parse(self):
        '''
        Gets coordinates of backbone
        '''
        # Computations
        num_models = 0
        for model in self.structure:
            if num_models > 0:
                print('Error, there are many models in this PDB file, while only one was expected. Have you cleaned it?')

            for chain in model:
                chain_id = chain.get_id() or 'A' # use A instead of None or empty string

                self.all_atoms_coords[chain_id] = []
                self.all_atoms_names[chain_id] = []
                self.all_atomic_groups[chain_id] = []

                self.res_atoms[chain_id] = []
                self.res_atoms_coords[chain_id] = []
                self.res[chain_id] = []
                self.res_seqnum[chain_id] = []
                self.chain_length[chain_id] = 0
                self.a_coords[chain_id] = []
                self.b_coords[chain_id] = []

                for residue in chain:

                    if is_aa(residue, standard=True): # Check if standard amino acid
                        resname = residue.get_resname()
                        respos = residue.get_id()[1]
                        self.res[chain_id].append(self.threetoone[resname]) # save which aa
                        self.res_seqnum[chain_id].append(respos) # save which pos
                        self.chain_length[chain_id] += 1 # add to the count

                        res_atoms_name, res_atoms_coords = [], [] # Temp arrays, refresh for each aa
                        for atom in residue:
                            atom_name = atom.get_name()

                            if atom_name in self.atom_ids:
                                xyz = atom.get_coord()

                                # save the residue-specific info
                                res_atoms_name.append(atom_name)
                                res_atoms_coords.append(xyz)

                                # add to ditionaries all the names and coordinates
                                self.all_atoms_names[chain_id].append(atom_name)
                                self.all_atoms_coords[chain_id].append(xyz)

                                # add the atomic group
                                atomic_group = self.atomic_groups[resname.upper()][atom_name]
                                self.all_atomic_groups[chain_id].append(atomic_group)

                                if atom_name == 'CA':
                                    self.a_coords[chain_id].append(xyz) # save alpha carbon pos

                                if atom_name == 'CB' or (atom_name == 'CA' and resname == 'GLY'):
                                    self.b_coords[chain_id].append(xyz) # save beta carbon pos


                        self.res_atoms[chain_id].append(res_atoms_name) # save atom names (grouped per aa)
                        self.res_atoms_coords[chain_id].append(np.array(res_atoms_coords)) # save atom coords (grouped per aa)

                self.all_atoms_coords[chain_id] = np.array(self.all_atoms_coords[chain_id]) # convert the atom coordinate list to numpy array
                self.length += self.chain_length[chain_id] # count all the amino acids

            num_models += 1


    def __get_seqres(self):
        '''
        Get the amino acids of the sequence actually used in the
        experiment, some of which may be missing in the structure.
        '''
        num_records = 0
        for record in SeqIO.parse(self.filepath, "pdb-seqres"): # have to use the non-cleaned file here
            # if num_records > 0:
            #     break
            seq = record.seq
            chain = record.annotations["chain"]
            self.seqres_seqnum[chain] = np.arange(1, len(seq)+1)
            self.seqres[chain] = list(seq)
            num_records += 1

    def __align_seqs(self):
        '''
        Some amino acids are often missing from the structure.
        By aligning the structure amino acids with those in the
        SEQRES portion I want to identify which ones are missing.
        These appear as gaps in the alignment.
        '''
        # assemble virtual fasta file
        for chain in self.res.keys():
            records = '>{}\n{}\n>{}\n{}'.format('res', ''.join(self.res[chain]),
                                                'seqres', ''.join(self.seqres[chain]))

            #turn string into a handle
            records_handle = io.StringIO(records)
            tempdata = records_handle.getvalue()

            # carry out the alignment
            muscle_cline = MuscleCommandline()
            stdout, stderr = muscle_cline(stdin=tempdata)

            # find the aligned amino acid sequence from the "structure" amino acid sequence
            with io.StringIO(stdout) as fasta:
                aln = SeqIO.parse(fasta, "fasta")

                # even though 'res' should be first, there is a Muscle bug that puts sequences out
                # of order. So iterate through, just to make sure.
                for entry in aln:
                    header = entry.description
                    if header == 'res':
                        aln_seq = str(entry.seq)
                        break

            # find positions for the gaps
            self.structure_gaps[chain] = []
            real_aa_counter = 0
            for pos_idx in range(0, len(aln_seq)):
                if aln_seq[pos_idx] == '-':
                    self.structure_gaps[chain].append(real_aa_counter)
                else:
                    real_aa_counter += 1


class PDBDistances(PDBTools):
    '''
    Compute various distance matrices.
    '''
    def __init__(self, pdb_obj, separation=1):
        super().__init__()
        self.separation = separation
        self.pdb_obj = pdb_obj

        # add in nan values where there are gaps in the structure
        self.__insert_gaps()

    def __insert_gaps(self):
        '''
        Go through all the data and insert gaps where
        amino acids are missing in the structure.
        '''
        self.pdb_obj.length = 0
        for chain in self.pdb_obj.res_atoms.keys():
            # simple lists
            self.pdb_obj.res_atoms[chain] = np.insert(self.pdb_obj.res_atoms[chain], self.pdb_obj.structure_gaps[chain], np.nan)
            self.pdb_obj.res_atoms_coords[chain] = np.insert(self.pdb_obj.res_atoms_coords[chain], self.pdb_obj.structure_gaps[chain], np.nan)
            self.pdb_obj.res[chain] = np.insert(self.pdb_obj.res[chain], self.pdb_obj.structure_gaps[chain], np.nan)

            # more complicated things, lists of arrays etc.
            self.pdb_obj.a_coords[chain] = np.insert(self.pdb_obj.a_coords[chain],
                                                    self.pdb_obj.structure_gaps[chain],
                                                    np.array([np.nan, np.nan, np.nan]),
                                                    axis=0)
            self.pdb_obj.b_coords[chain] = np.insert(self.pdb_obj.b_coords[chain],
                                                    self.pdb_obj.structure_gaps[chain],
                                                    np.array([np.nan, np.nan, np.nan]),
                                                    axis=0)

            self.pdb_obj.chain_length[chain] = self.pdb_obj.b_coords[chain].shape[0]
            self.pdb_obj.length += self.pdb_obj.chain_length[chain] # count all the amino acids

            ### Need to make this work for the three atom lists as well ###

    def intra_chain_pairwise_aa_distance_matrix(self):
        '''
        Fill a dictionary with pairwise distances of all amino acid residue pairs in a chain.
        This method uses `separation` which can be tuned accordingly.
        '''
        if not self.pdb_obj.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        feature_pdm_intra = {}
        self.aa = []
        for chain1 in self.pdb_obj.chain_length.keys():
            # calculate pairwise distances within chain A
            if chain1 == 'A':
                # Calculate pairwise interactions separately for each chain, sum the result
                for i, c1 in enumerate(self.pdb_obj.b_coords[chain1]):
                    self.aa.append(self.pdb_obj.seqres_seqnum[chain1][i]) # store the amino acid sequence number

                    for j, c2 in enumerate(self.pdb_obj.b_coords[chain1]):
                        if i+self.separation < j:
                            dist = self.calc_dist_3d(c1, c2)

                            if feature_pdm_intra.get(i+1) is None:
                                feature_pdm_intra[i+1] = {} # +1 to go from vector index to real seq index

                            if feature_pdm_intra.get(j+1) is None:
                                feature_pdm_intra[j+1] = {}

                            feature_pdm_intra[i+1][j+1] = dist
                            feature_pdm_intra[j+1][i+1] = dist

        # make data frame
        self.feature_pdm_intra = pd.DataFrame(feature_pdm_intra)
        self.feature_pdm_intra.sort_index(axis=0, inplace=True)
        self.feature_pdm_intra.sort_index(axis=1, inplace=True)


    def inter_chain_pairwise_aa_distance_matrix(self):
        '''
        Fill a dictionary with pairwise distances of all amino acid residue pairs between chains.
        '''
        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        self.feature_pdm_inter = {}
        self.aa = []
        chain1 = 'A'
        # Calculate pairwise distances between chain A and the others
        for chain2 in self.chain_length.keys():
            if chain1 != chain2:
                for i, c1 in enumerate(self.b_coords[chain1]):
                    for j, c2 in enumerate(self.b_coords[chain2]):

                        # the chains may differ in their starting positions, so I have to make sure I'm logging the correct contact here
                        if self.res_seqnum[chain1][i] == self.res_seqnum[chain2][j]:
                            j_adj = j
                        else:
                            j_adj = None
                            for idx, _ in enumerate(self.b_coords[chain1]):
                                if self.res_seqnum[chain1][idx] == self.res_seqnum[chain2][j]:
                                    j_adj = idx
                                    break
                            if j_adj is None:
                                print('this position does not exist in chain A')
                                continue

                        # if self.feature_pdm_intra.get(j_adj) is None: # guard against residues present in the other chains, but not in chain A
                        #     continue

                        if self.feature_pdm_inter.get(i) is None:
                            self.feature_pdm_inter[i] = {}

                        if self.feature_pdm_inter.get(j_adj) is None:
                            self.feature_pdm_inter[j_adj] = {}

                        if self.feature_pdm_inter[i].get(j_adj) is None:
                            self.feature_pdm_inter[i][j_adj] = float('Inf')
                            self.feature_pdm_inter[j_adj][i] = float('Inf')

                        dist = self.calc_dist_3d(c1, c2)
                        if dist < self.feature_pdm_inter[i][j_adj]: # update in case the interchain contacts are shorter than previously found ones
                            self.feature_pdm_inter[i][j_adj] = dist
                            self.feature_pdm_inter[j_adj][i] = dist

        # # make data frame
        self.feature_pdm_inter = pd.DataFrame(self.feature_pdm_inter)
        #
        # # change column and index names
        # self.dist_mat.columns = self.aa
        # self.dist_mat.index = self.aa
        # self.dist_mat.reset_index()
        #
        # # save to file
        # self.dist_mat.to_csv(join(folder_path, '{}_dist_mat.tsv'.format(basename(self.filename).replace('.pdb', ''))), sep='\t', index=True)


    def pairwise_aa_contact_matrix(self, mode='both'):
        '''
        Fill a dictionary with pairwise interactions of amino acid residue pairs.
        Dictionary is saved as (k,v) with (AA pair, nr_of_contacts).
        If the protein has multiple chains, nr_of_contacts will be all pairwise interactions
            internally within each chain, aggregated with all pairwise interactions between
            the chains.
        The mode parameter (intra, inter, both) determines what type of contacts are returned
        (inter-chain, intra-chain, or both)
        '''
        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        assert mode in ['intra', 'inter', 'both']

        # compute the distances
        self.intra_chain_pairwise_aa_distance_matrix()
        self.inter_chain_pairwise_aa_distance_matrix()

        # A list of all possible amino acid pairs. AC is the same as CA so any reversed duplicates are removed.
        aa_pairs = [''.join(i) for i in combinations_with_replacement(sorted(self.onetothree.keys()), 2)]

        self.feature_pcm = {c: 0 for c in aa_pairs}
        for i in self.feature_pdm_intra.keys():
            for j in self.feature_pdm_intra[i].keys():
                if i < j:
                    if mode == 'intra':
                        dist = self.feature_pdm_intra[i][j]
                    elif mode == 'inter':
                        dist = self.feature_pdm_inter[i][j]
                    elif mode == 'both':
                        dist = min(self.feature_pdm_intra[i][j], self.feature_pdm_inter[i][j])

                    if dist < self.aa_contact_dist:
                        res1 = self.res['A'][i]
                        res2 = self.res['A'][j]
                        self.feature_pcm[''.join(sorted(res1+res2))] += 1

        if not self.count_based_features:
            total = sum(self.feature_pcm.values(), 0.0) or 1
            self.feature_pcm = {k: v / total for k, v in self.feature_pcm.items()}



    def intra_chain_pairwise_atom_distance_matrix(self):
        '''
        '''
        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        # get all of the distances
        self.feature_atom_pdm_intra = self.calc_dist_vectorized_all_atom_pairs(self.all_atoms_coords['A'],
                                                         self.all_atoms_coords['A'])

        # turn it into a numpy array
        self.feature_atom_pdm_intra = pd.DataFrame(self.feature_atom_pdm_intra).to_numpy()

        # but I don't want contact within amino acids and not with neighbors, so create a mask for correcting the matrix
        atoms_to_remove = []
        aa_num = len(self.res_atoms_coords['A'])
        atom_counter = 0
        for i in range(0, aa_num):
            if i+1 >= aa_num:
                break

            atoms1 = list(range(atom_counter, atom_counter + len(self.res_atoms_coords['A'][i])))
            atoms2 = list(range(atoms1[-1] + 1, atoms1[-1] + 1 + len(self.res_atoms_coords['A'][i+1])))

            atom_counter = atoms1[-1] + 1
            atoms_to_remove.extend(itertools.product(atoms1 + atoms2, repeat=2))

        volume = np.ones(self.feature_atom_pdm_intra.shape) # Initialization of the mask
        volume[tuple(np.array(atoms_to_remove).T)] = np.nan # Convert flip values in mask according to atom positions
        self.feature_atom_pdm_intra = self.feature_atom_pdm_intra * volume # modify the distance values


    def inter_chain_pairwise_atom_distance_matrix(self):
        '''
        '''
        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        # get all of the distances
        self.feature_atom_pdm_inter = None
        chain1 = 'A'
        # Calculate pairwise distances between chain A and the others
        for chain2 in self.chain_length.keys():
            if chain1 != chain2:
                pdm_inter = self.calc_dist_vectorized_all_atom_pairs(self.all_atoms_coords[chain1],
                                                                    self.all_atoms_coords[chain2])
                if self.feature_atom_pdm_inter is None:
                    self.feature_atom_pdm_inter = pdm_inter
                else:
                    self.feature_atom_pdm_inter = np.minimum(self.feature_atom_pdm_inter, pdm_inter)

        #### How do I deal with different chain indexing? In case aa are missing ??? ####

        # turn it into a numpy array
        self.feature_atom_pdm_inter = pd.DataFrame(self.feature_atom_pdm_inter).to_numpy()


    def distance_to_water(self, trio_obj):
        '''
        Make use of triominoes to compute which atom "triangles"
        are on the protein outside, then compute each atoms minimum
        distance to one of those triangles. The "triangles" are
        identified by rolling a ball over the surface of the protein,
        so it is analagous to the proteins interaction with water.
        '''
        self.all_atoms_dist_from_surface = {}
        for chain in self.all_atoms_coords.keys():
            # compute the full distance matrix
            water_dist_mat = self.calc_dist_vectorized_all_atom_pairs(trio_obj.all_coords,
                                                            self.all_atoms_coords[chain])

            # how far away is the closest "water"?
            self.all_atoms_dist_from_surface[chain] = water_dist_mat.min(axis=1)



class PDBFeatures(PDBTools):
    """
    Protein Feature Calculations is a python program for calculating protein features on PDB files.

    The output consists of fixed-length feature vectors from parsed PDBs.

    Parameters
    ----------

    pdb_obj : object
        PDBParser object containing a parse PDB file
    aa_contact_dist : integer
        Cutoff distance, in Angstroms, for two amino acids to count as interacting, based on beta carbon position (default 6(Å))
    atom_contact_dist : integer
        Cutoff distance, in Angstroms, for two atoms acids to count as interacting (default 3(Å))
    separation : integer
        When set to 1, the program will not compare any two subsequent residues in the sequence (default 1)
    features : list
        List of features to output (default all features)
          Each feature is one column in the output TSV unless otherwise stated. The features are
            - FEATURE_CHAINS: number of chains
            - FEATURE_RESIDUES: number of residues
            - FEATURE_PDM: pairwise residue interactions (a pairwise distance matrix, 210 columns)
            - FEATURE_CO: relative and absolute contact order (2 columns)
            - FEATURE_ROG: radius of gyration.
            - FEATURE_SA: atomic groups on the surface (13 columns)
            - FEATURE_PHIPSI: Phi and psi torsion angles (11 columns)
    count_base_features : boolean
        Whether the featuers could be count-based or not (in which case they are frequency-based) (default False)
    output_dir : string
        Where features should be saved to (default current folder)

    Running the program
    -------------------

    (1) To calculate all features
        pdb_features = PDBFeatures()
        pdb_features.parse_pdb(pdb_file)
        pdb_features.calculate_features()
        pdb_features.save_csv()

    (2) To calculate specified features
        pdb_fetures = PDBFeatures(features=[PDBFeatures.FEATURE_PDM])
        pdb_features.parse_pdb(pdb_file)
        pdb_features.calculate_features()
        pdb_features.save_csv()

    (3) It is also possible to manually calculate features as
        pdb_features = PDBFeatures(features=[PDBFeatures.FEATURE.PDM])
        pdb_features.parse_pdb(pdb_file)
        pdb_features.pairwise_aa_contact_matrix()
        pdb_features.save_csv()
    """
    FEATURE_ID_CHAINS = '#Chains'
    FEATURE_ID_RESIDUES = '#Residues'
    FEATURE_ID_PDM ='Pairwise distance matrix'
    FEATURE_ID_CO = 'Contact order'
    FEATURE_ID_ROG = 'Radius of gyration'
    FEATURE_ID_SA = 'Surface atoms'
    FEATURE_ID_PHIPSI = 'Phi/Psi torsion angles'

    def __init__(self,
                pdb_obj,
                aa_contact_dist=6,
                atom_contact_dist=3,
                separation=1,
                features=[FEATURE_ID_CHAINS,
                          FEATURE_ID_RESIDUES,
                          FEATURE_ID_PDM,
                          FEATURE_ID_CO,
                          FEATURE_ID_ROG,
                          FEATURE_ID_SA,
                          FEATURE_ID_PHIPSI,
                          ],
                count_based_features=False, # Add to readme: if (relevant) features should be stored in a count-based or frequency-based manner
                output_dir=abspath('.')):
        super().__init__()

        self.pdb_obj = pdb_obj

        # self.topts = self.__get_topts()
        self.__atomic_groups = None
        self.__phipsi_angles = None
        self.aa_contact_dist = aa_contact_dist
        self.atom_contact_dist = atom_contact_dist
        self.separation = separation
        self.features = features
        self.feature_map = {
            self.FEATURE_ID_CHAINS: self.calculate_feature_chains,
            self.FEATURE_ID_RESIDUES: self.calculate_feature_residues,
            self.FEATURE_ID_PDM: self.pairwise_aa_contact_matrix,
            self.FEATURE_ID_CO: self.contact_order_vectorized,
            self.FEATURE_ID_ROG: self.radius_of_gyration,
            self.FEATURE_ID_SA: self.surface_atoms,
            self.FEATURE_ID_PHIPSI: self.phipsi_angles,
        }
        self.count_based_features = count_based_features
        self.output_dir = output_dir

        # setup data variables
        self.filename = self.pdb_obj.filename
        self.all_atoms_names = self.pdb_obj.all_atoms_names # dictionary with chain_id keys holding list with all atom names
        self.all_atoms_coords = self.pdb_obj.all_atoms_coords # dictionary with chain_id keys holding list with all atom coordinates
        self.all_atomic_groups = self.pdb_obj.all_atomic_groups # dictionary with chain_id keys holding list with all atomic groups (13 possible)

        self.res_atoms = self.pdb_obj.res_atoms # dictionary with chain_id keys holding list of lists with the atom names for each amino acid
        self.res_atoms_coords = self.pdb_obj.res_atoms_coords # dictionary with chain_id keys holding list of lists with the atom coordinates for each amino acid
        self.residues = self.pdb_obj.res # dictionary with chain_id as keys holding list of the one-letter amino acids as they occur
        self.res_seqnum = self.pdb_obj.res_seqnum # dictionary with chain_id as keys holding list of amino acid numbers (position in protein seq)
        self.chain_length = self.pdb_obj.chain_length # dictionary with chain_id as keys holding the chain length as an integer
        self.a_coords = self.pdb_obj.a_coords # dictionary with chain_id keys holding list alpha carbon coordinates for each amino acid
        self.b_coords = self.pdb_obj.b_coords # dictionary with chain_id keys holding list beta carbon coordinates for each amino acid

        self.type = self.pdb_obj.type
        self.ec_number = self.pdb_obj.ec_number
        self.length = self.pdb_obj.length

        # If including more features, add corresponding dictionary here
        self.feature_residues = {}
        self.feature_chains = {}
        self.feature_pdm = {}
        self.feature_relative_co = {}
        self.feature_abs_co = {}
        self.feature_rog = {}
        self.feature_atomic_groups = {}
        self.feature_triangle_groups = {}
        self.feature_phipsi_angles = {}

        ### Add feature that is buriedness ###
        ### Add feature whether a contact is local (what's the cutoff for this?) or not ###
        ### Change contact features to incorporate a few distance bins ###
        ### Also include voronoid once that is done ###

    def calculate_features(self):
        '''
        This will calculate the features specified in self.features
        '''
        for feature_id in self.features:
            if feature_id in self.feature_map:
                self.feature_map[feature_id]()


    def calculate_feature_chains(self):
        self.feature_chains['#Chains'] = len(self.chain_length)


    def calculate_feature_residues(self):
        self.feature_residues['#Residues'] = self.length


    def contact_order(self):
        '''
        Calculates the Absolute Contact Order (Abs_CO) and Relative Contact Order (CO) of a protein.
        Formulas from
            Ivankov, D. N., Garbuzynskiy, S. O., Alm, E., Plaxco, K. W., Baker, D., & Finkelstein, A. V. (2003).
            Contact order revisited: influence of protein size on the folding rate. Protein science, 12(9), 2057-2062.
        If the protein has multiple chains, calculate `abs_co` and `relative_co` for each, and average the results.
        Accessed with self.abs_co and self.relative_co
        '''

        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        abs_co, relative_co = [], []
        for chain, _ in self.chain_length.items():
            nr_contacts = 0
            seq_dist = 0
            for i in range(len(self.residues[chain])):
                for j in range(i+1, len(self.residues[chain])):
                    stop = False
                    for c1 in self.res_atoms_coords[chain][i]:
                        for c2 in self.res_atoms_coords[chain][j]:
                            dist = self.calc_dist_3d(c1, c2)
                            if dist < self.aa_contact_dist:
                                nr_contacts += 1
                                seq_dist += (self.res_seqnum[j] - self.res_seqnum[i])
                                stop = True
                            if stop: break
                        if stop: break


            abs_co.append(seq_dist / nr_contacts)
            relative_co.append(seq_dist / (nr_contacts * len(self.residues[chain])))
        self.abs_co = np.mean(abs_co)
        self.relative_co = np.mean(relative_co)
        self.feature_relative_co['Relative CO'] = np.mean(relative_co)
        self.feature_abs_co['Abs CO'] = np.mean(abs_co)


    def contact_order_vectorized(self):
        '''
        New version of contact order that makes use of vectorization
        to speed things up.
        '''

        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        abs_co, relative_co = [], []
        for chain, _ in self.chain_length.items():
            nr_contacts = 0
            seq_dist = 0
            for i in range(len(self.residues[chain])):
                for j in range(i+1, len(self.residues[chain])):
                    dist = self.calc_dist_vectorized_all_atom_pairs(self.res_atoms_coords[chain][i], self.res_atoms_coords[chain][j])
                    if dist < self.aa_contact_dist:
                        nr_contacts += 1
                        seq_dist += (self.res_seqnum[chain][j] - self.res_seqnum[chain][i])

            abs_co.append(seq_dist / nr_contacts)
            relative_co.append(seq_dist / (nr_contacts * len(self.residues[chain])))
        self.abs_co = np.mean(abs_co)
        self.relative_co = np.mean(relative_co)
        self.feature_relative_co['Relative CO'] = np.mean(relative_co)
        self.feature_abs_co['Abs CO'] = np.mean(abs_co)


    def radius_of_gyration(self):
        '''
        Calculates radius of gyration for a protein. From
            Lobanov, M. Y., Bogatyreva, N. S., & Galzitskaya, O. V. (2008).
            Radius of gyration as an indicator of protein structure compactness. Molecular Biology, 42(4), 623-628.
        If the protein has multiple chains, treat it as one assembly and calculate
            RoG for the entire protein.
        Accessed with self.rog.
        '''

        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        nr_atoms = 0
        sq_sum = 0
        centroid_x, centroid_y, centroid_z = 0,0,0 # Center of mass
        for chain, _ in self.chain_length.items():
            for i in range(len(self.residues[chain])):
                for j in self.res_atoms_coords[chain][i]:
                    nr_atoms += 1
                    centroid_x += j[0]
                    centroid_y += j[1]
                    centroid_z += j[2]
        centroid_x /= nr_atoms
        centroid_y /= nr_atoms
        centroid_z /= nr_atoms

        for chain, _ in self.chain_length.items():
            for i in range(len(self.residues[chain])):
                for j in self.res_atoms_coords[chain][i]:
                    dist = self.calc_dist_3d(j, (centroid_x,centroid_y,centroid_z))
                    sq_sum += dist * dist
        self.rog = math.sqrt(sq_sum / nr_atoms)
        self.feature_rog['Radius of Gyration'] = math.sqrt(sq_sum / nr_atoms)


    # def surface_atoms(self):
    #     '''
    #     Save the output from __load_triominoes() to a feature dictionary.
    #     See documentation in __load_triominoes() for more details.
    #     '''
    #     if not self.__atomic_groups:
    #         self.__atomic_groups = self.__load_triominoes()
    #     name = basename(self.filename)[:-4]
    #     uniprotid = name.split('_')[0]
    #     self.feature_surface_atoms = self.__atomic_groups[uniprotid]
    #
    #     if not self.count_based_features:
    #         total = sum(self.feature_surface_atoms.values(), 0.0) or 1
    #         self.feature_surface_atoms = {k: v / total for k, v in self.feature_surface_atoms.items()}

    def surface_atoms(self):
        '''
        Save the output from __load_triominoes() to a feature dictionary.
        See documentation in __load_triominoes() for more details.
        '''
        self.feature_atomic_groups, self.feature_triangle_groups = self.__load_triominoes()

        if not self.count_based_features:
            total = sum(self.feature_atomic_groups.values(), 0.0) or 1
            self.feature_atomic_groups = {k: v / total for k, v in self.feature_atomic_groups.items()}

            total = sum(self.feature_triangle_groups.values(), 0.0) or 1
            self.feature_triangle_groups = {k: v / total for k, v in self.feature_triangle_groups.items()}

    def phipsi_angles(self):
        '''
        For each enzyme there is a corresponding dict entry which has stored all phi and psi angles in the enzyme. For each such phi/psi pair, find
        the corresponding basin it belongs to. The 11 basins are defined and used in
            Chellapa, G. D., & Rose, G. D. (2012). Reducing the dimensionality of the protein‐folding
            search problem. Protein Science, 21(8), 1231-1240.
        Use Euclidean distance to calculate the distance. Store a count for each basin.
        '''
        phipsi_basins = {'A':(-62,-42),'B':(-120,135),'D':(-134,70),'G':(-93,95),'L':(51,42),'P':(-64,139),
                         'R':(-68,-18),'T':(55,-129),'U':(82,-3),'V':(-93,2),'Y':(77,-171)}

        if not self.__phipsi_angles:
            self.__phipsi_angles = self.__load_phipsi()
        name = self.filename
        uniprotid = name.split('_')[0]
        phipsi_pairs = self.__phipsi_angles[uniprotid]

        # The 11 basins
        self.feature_phipsi_angles = {c: 0 for c in list(phipsi_basins.keys())}
        min_dist = float('Inf')
        min_basin = ''

        for phipsi in phipsi_pairs:
            for basin,loc in phipsi_basins.items():
                d = self.__calc_dist_2d(phipsi,loc)
                if d < min_dist:
                    min_dist = d
                    min_basin = basin
            self.feature_phipsi_angles[min_basin] += 1

        if not self.count_based_features:
            total = sum(self.feature_phipsi_angles.values(), 0.0) or 1
            self.feature_phipsi_angles = {k: v / total for k, v in self.feature_phipsi_angles.items()}



    def save_csv(self):
            '''
            Saves all the calculated features into a TSV.
            '''
            # get basic info
            name = self.filename
            pdb_id = name.split('_')[0]

            # save info about the protein
            save_path = join(self.output_dir, 'features_basic.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format('pdb_id', 'type', 'ec', 'num_residues', 'num_chains'))
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(pdb_id,
                                                      self.type,
                                                      self.ec_number,
                                                      self.feature_residues['#Residues'],
                                                      self.feature_chains['#Chains']))

            # save info about contact order
            save_path = join(self.output_dir, 'features_contact_order.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\t{}\n'.format('pdb_id', 'absolute_co', 'relative_co'))
                f.write('{}\t{}\t{}\n'.format(pdb_id,
                                              self.feature_abs_co['Abs CO'],
                                              self.feature_relative_co['Relative CO']))

            # save info about radius of gyration
            save_path = join(self.output_dir, 'features_radius_of_gyration.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\n'.format('pdb_id', '\t'.join(list(sorted(self.feature_rog.keys())))))
                f.write('{}\t{}\n'.format(pdb_id, '\t'.join(
                    [str(self.feature_rog[k]) for k in sorted(self.feature_rog.keys())])))

            # save info about surface atoms
            save_path = join(self.output_dir, 'features_surface_atoms.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\n'.format('pdb_id', '\t'.join(list(sorted(self.feature_atomic_groups.keys())))))
                f.write('{}\t{}\n'.format(pdb_id, '\t'.join(
                    [str(self.feature_atomic_groups[k]) for k in sorted(self.feature_atomic_groups.keys())])))

            # save info about surface triangles
            save_path = join(self.output_dir, 'features_surface_triangles.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\n'.format('pdb_id', '\t'.join(list(sorted(self.feature_triangle_groups.keys())))))
                f.write('{}\t{}\n'.format(pdb_id, '\t'.join(
                    [str(self.feature_triangle_groups[k]) for k in sorted(self.feature_triangle_groups.keys())])))

            # save info about the angles
            save_path = join(self.output_dir, 'features_angles.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\n'.format('pdb_id', '\t'.join(list(sorted(self.feature_phipsi_angles.keys())))))
                f.write('{}\t{}\n'.format(pdb_id, '\t'.join(
                    [str(self.feature_phipsi_angles[k]) for k in sorted(self.feature_phipsi_angles.keys())])))

            # save info about pairwise contacts
            save_path = join(self.output_dir, 'features_pairwise_contacts.tsv')
            newfile = os.path.isfile(save_path)
            with open(save_path, 'a') as f:
                if not newfile:
                    f.write('{}\t{}\n'.format('pdb_id', '\t'.join(list(sorted(self.feature_pdm.keys())))))
                f.write('{}\t{}\n'.format(pdb_id, '\t'.join(
                    [str(self.feature_pdm[k]) for k in sorted(self.feature_pdm.keys())])))


    def __get_topts(self):
        '''
        Fetch the optimal catalytic temperatures from 'cleaned_enzyme_topts_with_structures.fasta', which
        can be found in data/sequences.
        '''
        with open(join(seqdir, 'cleaned_enzyme_topts_with_structures.fasta'), 'r') as f:
            topts = defaultdict(float)
            for row in f:
                if row[0] == '>':
                    uniprotid = row.split(' ')[0][1:]
                    topt = row.strip().split('=')[-1]
                    topts[uniprotid] = topt
        return topts


    # def __load_triominoes(self):
    #     '''
    #     This helper function loads the triominoes_output folder, and saves, for each enzyme,
    #     its corresponding surface atoms to a dict.
    #     The files come from running the .sh script in the Triominoes folder. The script file runs triominoes.c for each PDB file,
    #     program by Graham J.L Kemp, used in
    #         Mehio, W., Kemp, G. J., Taylor, P., & Walkinshaw, M. D. (2010). Identification of protein binding
    #         surfaces using surface triplet propensities. Bioinformatics, 26(20), 2549-2555.
    #     A probe of radius 1.5Å is rolled over the surface of a protein, and the program searches for triplets of atomic group that
    #     the probe can touch simultaneously. 13 such atomic groups exist.
    #     '''
    #     ATOM_GROUPS = ['C3H0', 'C3H1', 'C4H1', 'C4H2', 'C4H3', 'N3H0', 'N3H1', 'N3H2', 'N4H3', 'O1H0', 'O2H1', 'S2H0', 'S2H1']
    #     atomic_groups_per_enzyme = {}
    #     for d, _subdirs, files in os.walk(triominoes_output):
    #         for f in files:
    #             if f.endswith('.txt'):
    #                 with open(join(d, f), 'r') as triominoes_file:
    #                     f = f.split('_')[1]
    #                     lines = [line.strip().split(' ') for line in triominoes_file]
    #                     ag = {atom_group: int(count) for count, atom_group in lines if atom_group != 'UNKN'}
    #                     for atom_group in ATOM_GROUPS:
    #                         if atom_group not in ag:
    #                             ag[atom_group] = 0
    #                     atomic_groups_per_enzyme[f] = ag
    #     return atomic_groups_per_enzyme


    def __load_triominoes(self):
        '''
        This helper function loads the triominoes_output folder, and saves, for each enzyme,
        its corresponding surface atoms to a dict.
        The files come from running the .sh script in the Triominoes folder. The script file runs triominoes.c for each PDB file,
        program by Graham J.L Kemp, used in
            Mehio, W., Kemp, G. J., Taylor, P., & Walkinshaw, M. D. (2010). Identification of protein binding
            surfaces using surface triplet propensities. Bioinformatics, 26(20), 2549-2555.
        A probe of radius 1.5Å is rolled over the surface of a protein, and the program searches for triplets of atomic group that
        the probe can touch simultaneously. 13 such atomic groups exist.
        '''
        ATOM_GROUPS = ['C3H0', 'C3H1', 'C4H1', 'C4H2', 'C4H3', 'N3H0', 'N3H1', 'N3H2', 'N4H3', 'O1H0', 'O2H1', 'S2H0', 'S2H1']

        # count each individual atom group
        ag_data = {k:0 for k in ATOM_GROUPS}

        # count the unique "triangles" of atom groups interacting
        tr_data = {k:0 for k in ['_'.join(sorted(c)) for c in combinations_with_replacement(ATOM_GROUPS, 3)]}

        with open(join('../data/pdb_features/triominoes_output/', '{}_full.txt'.format(self.pdb_id)), 'r') as f:
            data = f.read()

            groups = [s.strip().split(' ')[1] for s in set(re.findall(' [0-9]+ [A-Z0-9]+', data))]
            for gr in groups:
                if gr != 'UNKN':
                    ag_data[gr] += 1

            triangles = re.findall(' [0-9]+ [A-Z0-9]+ [0-9]+ [A-Z0-9]+ [0-9]+ [A-Z0-9]+', data)
            for tr in set(triangles):
                _, one, _, two, _, three = tr.strip().split(' ')
                if not 'UNKN' in [one, two, three]:
                    canonical_triangle = '_'.join(sorted([one, two, three]))
                    tr_data[canonical_triangle] += 1

        return ag_data, tr_data


    def __load_phipsi(self):
        '''
        This helper function loads the phipsi_output folder and saves, for each enzyme, its corresponding phi/psi torsion
        angles to a dictionary.
        The files come from running the .sh script in the phipsi folder. The script file runs phipsi.c for each PDB file,
        program by Graham J.L Kemp. The C program calculated the phi and psi torsion angles for each residue in a protein.
        '''
        phi_psi_per_enzyme = {}
        for d, _subdirs, files in os.walk(phipsi_output):
            for f in files:
                if f.endswith('.txt'):
                    with open(join(d, f), 'r') as phipsi_file:
                        f = f.split('_')[1]
                        lines = [line.strip().split(',') for line in phipsi_file]
                        phipsi = [(float(phi),float(psi)) for _,_,_,phi,psi in lines]
                        phi_psi_per_enzyme[f] = phipsi
        return phi_psi_per_enzyme
