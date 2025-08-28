import os

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt

def fix_four_bonded_nitrogen_charges(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.RWMol(mol) # Change immutable mol object to mutable mol object
    to_fix = []

    # Iterate over all atoms of the molecule
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue # Skip if atom is not nitrogen

        o_double = None # Stores double bonded O in N=O if found
        o_single = None # Stores isolated single bonded O in N-O if found

        # Iterate over neighbors of nitrogen atoms
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 8:
                continue # Skip if neighbor is not oxygen

            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())

            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                o_double = neighbor # N=O, nitrogen double bonded to oxygen 
            elif bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                # Get neighbors of oxygen in N-O structure 
                o_neighbors = [nbr for nbr in neighbor.GetNeighbors()\
                                if nbr.GetIdx() != atom.GetIdx()]
                # Check if single-bonded oxygen in N-O structure has
                # no other neighbors (isolated O)
                o_has_no_other_neighbors = len(o_neighbors) == 0

                if o_has_no_other_neighbors:
                    o_single = neighbor # Found the O in N-O structure
        
        # If nitrogen has both N=O and N-O (N(=O)O type structure), mark for fixing
        if o_double != None and o_single != None:
            to_fix.append((atom.GetIdx(), o_single.GetIdx()))

    # Apply formal charges to the nitrogen (+1) and oxygen (-1)
    for n_idx, o_idx in to_fix:
        mol.GetAtomWithIdx(n_idx).SetFormalCharge(+1)
        mol.GetAtomWithIdx(o_idx).SetFormalCharge(-1)

    # Return the corrected immutable molecule
    return mol.GetMol()

class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
            if self.remove_h:
                mol = Chem.RemoveHs(mol, sanitize=False)
            mol = fix_four_bonded_nitrogen_charges(mol)

        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")
