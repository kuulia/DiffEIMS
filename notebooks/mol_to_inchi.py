"""Print the InChI of a MOL or SDF file, e.g. one downloaded from the NIST WebBook.

    python mol_to_inchi.py nist_mols/LDY-18937.MOL
    python mol_to_inchi.py nist_mols/LDY-18937.SDF

An SDF with several records prints one InChI per record.
"""
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def read_mols(path):
    # NIST's MOL and SDF exports are the same text: each record ends in '$$$$' with
    # no 'M  END' line, which RDKit's file readers reject. Split on '$$$$' and
    # restore 'M  END' so each record parses as a mol block.
    with open(path) as f:
        text = f.read().replace('\r\n', '\n')
    blocks = [b for b in text.split('$$$$') if b.strip()]
    return [Chem.MolFromMolBlock(b if 'M  END' in b else b.rstrip('\n') + '\nM  END\n')
            for b in blocks]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    mols = read_mols(sys.argv[1])
    if not mols or any(m is None for m in mols):
        sys.exit(f'could not parse a structure from {sys.argv[1]}')
    for mol in mols:
        print(Chem.MolToInchi(mol))
