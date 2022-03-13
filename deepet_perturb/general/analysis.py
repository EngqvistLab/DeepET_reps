from collections import Counter
from typing import Iterable

import h5py
from goatools.obo_parser import GODag
from goatools.mapslim import mapslim
import numpy as np
import pandas as pd
from scipy import stats


def coverage_fract_in_window(zscores: np.ndarray, start: int, end: int,
                             zscore_cutoff: float=1) -> float:
    """
    Return the fraction of positions inside the given window
    that have an abs(zscore) >= the `zscore_cutoff`.

    Start and end given as Python slice indices (i.e. inclusive start, exclusive end)
    If start == end, return 0
    """
    if start == end:
        return 0
    n_above_cutoff = np.sum(np.abs(zscores[start:end]) >= zscore_cutoff)
    return n_above_cutoff / (end - start)


def mean_coverage_outside_window(zscores: np.ndarray, start: int, end: int,
                                 zscore_cutoff: float=1) -> float:
    """
    Return the fraction of positions outside the given window
    that have an abs(zscore) >= the `zscore_cutoff`,
    averaged by dividing the total coverage count by the size of the window.

    Equivalent to exhaustively averaging fractions from random samples of same-length windows.

    Start and end given as Python slice indices (i.e. inclusive start, exclusive end)
    If start == end, return fraction covered in entire sequence.
    """
    if start == end:
        return coverage_fract_in_window(zscores, start = 0, end = zscores.shape[0],
                                        zscore_cutoff = zscore_cutoff)
    n_above_cutoff_outside = (
         np.sum(np.abs(zscores[:start]) >= zscore_cutoff) +
         np.sum(np.abs(zscores[end:]) >= zscore_cutoff)
    )
    win_len = end - start
    n_segments = (zscores.shape[0] - win_len) / win_len
    mean_n_above_cutoff = n_above_cutoff_outside / n_segments
    return mean_n_above_cutoff / win_len


def test_coverage_fract():
    zscores = np.array([1, -2, 1, 2, 1, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    window = (0, 5)
    win_len = window[1] - window[0]
    expected_fract_in_window = 2 / win_len

    outside = np.array([0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    n_segments = 13 / 5
    expected_fract_outside_window = (2 / n_segments) / win_len

    assert np.isclose(
        coverage_fract_in_window(zscores, start=window[0], end=window[1], zscore_cutoff=2),
        expected_fract_in_window
    )
    assert np.isclose(
        mean_coverage_outside_window(zscores, start=window[0], end=window[1], zscore_cutoff=2),
        expected_fract_outside_window
    )
    assert np.isclose(coverage_fract_in_window(zscores, start=0, end=0, zscore_cutoff=2), 0)
    assert np.isclose(
        mean_coverage_outside_window(zscores, start=0, end=0, zscore_cutoff=2),
        (np.abs(zscores) >= 2).sum() / zscores.shape[0]
    )


def get_significant_aa_count(seq: str, zscores: np.ndarray, zscore_cutoff: float = 1) -> pd.DataFrame:
    """
    Return the count of each amino acid type
    in positions that have an abs(zscore) >= the `zscore_cutoff`
    """
    residues = [seq[i] for i in np.nonzero(np.abs(zscores) >= zscore_cutoff)[0]]
    aa_counts = pd.DataFrame.from_records(
        list(Counter(residues).items()),
        columns = ['aa', 'count']
    )
    return aa_counts


def test_get_significant_aa_count():
    seq = 'MUPPETSMUPPETS'
    zscores = np.array([2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2])
    expected_counts = pd.DataFrame.from_records(
        [('M', 1),
         ('U', 1),
         ('P', 2),
         ('T', 1),
         ('S', 1)],
        columns=['aa', 'count']
    ).sort_values('aa', ignore_index=True)

    counts = get_significant_aa_count(seq, zscores, zscore_cutoff=2)
    counts = counts.sort_values('aa', ignore_index=True)  # make comparable

    assert expected_counts.compare(counts).empty


def hypergeom_test_aa_counts(aa_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Performs two one-sided hypergeometric tests
    to assess under- and overrepresentaiton.

    Expects colums: aa, count, background_count
    Returns columns: ['aa', 'pval_under', 'pval_over']
    """
    total_count = aa_counts['count'].sum()
    total_background_count = aa_counts['background_count'].sum()
    aa_pval = []
    for _, row in aa_counts.iterrows():
        distrib = stats.hypergeom(M=total_background_count,
                                  n=row['background_count'],
                                  N=total_count)
        pval_under = distrib.cdf(row['count'])
        pval_over = distrib.sf(row['count'] - 1)
        aa_pval.append((row['aa'], pval_under, pval_over))
    return pd.DataFrame.from_records(aa_pval,
                                     columns=['aa', 'pval_under', 'pval_over'])


def test_hypergeom_test_aa_counts():
    aa_counts = pd.DataFrame.from_records(
        [('A', 80, 2000),
         ('B', 10, 2000),
         ('C', 10, 2000)],
        columns=['aa', 'count', 'background_count']
    )
    expected_pvals = pd.DataFrame.from_records(
        [('A', 1.000000e+00, 5.449704e-22),
         ('B', 4.300177e-08, 1.000000e+00),
         ('C', 4.300177e-08, 1.000000e+00)],
        columns=['aa', 'pval_under', 'pval_over']
    )
    sample_ratio_signif_aa = hypergeom_test_aa_counts(aa_counts)
    assert np.allclose(expected_pvals[['pval_under', 'pval_over']].values,
                       sample_ratio_signif_aa[['pval_under', 'pval_over']].values)


def get_goslims_for_ids(go_ids: list, goslim_dag: GODag, go_dag: GODag) -> pd.DataFrame:
    """
    Map given GO terms to their respective GO slim terms in the provided graph (DAG)
    """
    aspect_acronym = {'molecular_function': 'MF', 'biological_process': 'BP', 'cellular_component': 'CC'}
    go_slim_direct_ancestors = [
        mapslim(term, go_dag, goslim_dag)[0] for term in go_ids
    ]
    go_slim_terms = list(set.union(*go_slim_direct_ancestors))
    go_slim_terms = pd.DataFrame.from_records(
        [(go_id,
          aspect_acronym[goslim_dag[go_id].namespace],
          goslim_dag[go_id].name)
         for go_id in go_slim_terms],
        columns=['go_id', 'aspect', 'go_term']
    )
    return go_slim_terms


def jaccard_similarity(a: Iterable, b: Iterable) -> float:
    a = set(list(a))
    b = set(list(b))
    return len(a.intersection(b)) / len(a.union(b))


def load_occlusion_profiles_as_zscore_df(store_fname: str) -> pd.DataFrame:
    zscores = []
    with h5py.File(store_fname, 'r') as store:
        for uniprot_ac in store.keys():
            zscores.append((uniprot_ac, np.array(store[uniprot_ac]['z_scores'],
                                                 dtype=np.float32)))
    zscores = pd.DataFrame(zscores, columns=['uniprot_ac', 'zscores'])
    return zscores
