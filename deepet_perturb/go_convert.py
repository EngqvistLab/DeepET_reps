#!/usr/bin/env python3
"""
Simple GO OBO -> CSV converter, based on the release:
http://release.geneontology.org/2021-02-01/ontology/index.html

and using goatools==1.0.15

Note: The goatools internal representation is more likely to change over time
than the GO ontology itself.
"""
import argparse
import os
import sys

from goatools.obo_parser import GODag
import pandas as pd


def main():
    args = get_args()
    obo_graph = GODag(args.input)

    field_names = ['id', 'name', 'namespace', 'parents', 'children',
                   'level', 'depth', 'is_obsolete', 'alt_ids']
    entries = []
    for term in obo_graph.values():
        children_ids = [child.item_id for child in term.children]

        if term._parents:
            parent_list_as_str = ','.join(sorted(term._parents))
        else:
            parent_list_as_str

        if children_ids:
            children_list_as_str = ','.join(sorted(children_ids))
        else:
            children_list_as_str = ''

        if term.alt_ids:
            alternate_id_list_as_str =','.join(sorted(term.alt_ids))
        else:
            alternate_id_list_as_str = ''

        entries.append(
            (term.item_id, term.name, term.namespace,
             parent_list_as_str, children_list_as_str,
             term.level, term.depth, term.is_obsolete,
             alternate_id_list_as_str)
        )

    entries = pd.DataFrame.from_records(entries, columns=field_names)
    entries.to_csv(args.output, index=False)


def get_args():
    parser = argparse.ArgumentParser(
        description='Convert Gene Ontology OBO file to a CSV.'
                    'Requires libraries: pandas, goatools==1.0.15'
    )
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        type=str,
                        help='Input OBO file')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        type=str,
                        help='Output CSV file')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('ERROR: Input file does not exist at ' + args.input)
        sys.exit(1)
    return args


if __name__ == '__main__':
    main()
