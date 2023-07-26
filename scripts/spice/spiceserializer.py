import h5py
import jax
import numpy as onp
import time
import json
import os
import sys
import tqdm

class SPICESerializer:

    def __init__(self, data_path, out_prefix, train_ratio, test_ratio, seed=2666, max_atoms=96, dist_cutoff=10, max_edges=3312):
        self.data = h5py.File(data_path, 'r')
        self.names = list(self.data.keys())
        key = jax.random.PRNGKey(seed)
        self.max_atoms = max_atoms
        self.dist_cutoff = dist_cutoff
        self.max_edges = max_edges
        self.SUBSET_MAP = {"SPICE Dipeptides Single Points Dataset v1.2" : 0, "SPICE Solvated Amino Acids Single Points Dataset v1.1" : 1, "SPICE DES370K Single Points Dataset v1.0" : 2, "SPICE DES370K Single Points Dataset Supplement v1.0" : 2, "SPICE DES Monomers Single Points Dataset v1.1" : 3, "SPICE PubChem Set 1 Single Points Dataset v1.2": 4, "SPICE PubChem Set 2 Single Points Dataset v1.2" : 4, "SPICE PubChem Set 3 Single Points Dataset v1.2" : 4, "SPICE PubChem Set 4 Single Points Dataset v1.2" : 4, "SPICE PubChem Set 5 Single Points Dataset v1.2" : 4, "SPICE PubChem Set 6 Single Points Dataset v1.2" : 4, "SPICE Ion Pairs Single Points Dataset v1.1" : 5} 
        self.test_names, self.train_names, self.val_names = self._split(key, train_ratio, test_ratio)
        self._make_npz(self.train_names, out_prefix + "spice_train")
        self._make_npz(self.test_names, out_prefix + "spice_test")
        self._make_npz(self.val_names, out_prefix + "spice_val")

    def _split(self, key, train_ratio, test_ratio):
        n_samples = len(self.names)
        split_idxs = jax.random.permutation(key, n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = int(train_ratio * n_samples)
        test_names = [self.names[i] for i in split_idxs[:n_test]]
        train_names = [self.names[i] for i in split_idxs[n_test:n_test + n_train]]
        val_names = [self.names[i] for i in split_idxs[n_test + n_train:]]
        return test_names, train_names, val_names

    def _make_npz(self, names, out_path):
        all_pos = []
        all_atom_nums = []
        # all_total_energies = []
        all_form_energies = []
        all_grads = []
        all_subsets = []
        all_names = []
        all_edges = []
        all_num_nodes = []
        all_num_edges = []
        for name in tqdm.tqdm(names):
            atom_nums = onp.array(self.data[name]['atomic_numbers'], onp.uint8)
            if len(atom_nums) > self.max_atoms:
                print("Skipping: ", name)
                continue
            pos_arr = self.data[name]['conformations']
            grads_arr = self.data[name]['dft_total_gradient']
            # total_energy_arr = self.data[name]['dft_total_energy']
            form_energy_arr = self.data[name]['formation_energy']
            num_nodes = len(atom_nums)
            pad_num = self.max_atoms - num_nodes
            padded_atom_nums = onp.pad(atom_nums, (0, pad_num))
            padded_pos = onp.pad(pos_arr, ((0, 0), (0, pad_num), (0, 0)))
            padded_grads = onp.pad(grads_arr, ((0, 0), (0, pad_num), (0, 0)))
            edges, num_edges = self.batch_radius_graph(pos_arr, self.dist_cutoff, self.max_edges)
            all_atom_nums.append([padded_atom_nums for conf in range(len(pos_arr))])
            all_subsets.append([self.SUBSET_MAP[self.data[name]['subset'][0]] for conf in range(len(pos_arr))])
            all_names.append([name for conf in range(len(pos_arr))])
            # all_total_energies.append(total_energy_arr)
            all_form_energies.append(form_energy_arr)
            all_grads.append(padded_grads)
            all_pos.append(padded_pos)
            all_edges.append(edges)
            all_num_nodes.append([num_nodes for conf in range(len(pos_arr))])
            all_num_edges.append(num_edges)
        print(all_num_nodes)
        print(all_num_edges)
        onp.savez(out_path, atomic_numbers=np.concatenate(all_atom_nums), formation_energy=np.concatenate(all_form_energies), forces=-np.concatenate(all_grads), pos=np.concatenate(all_pos), names=np.concatenate(all_names), subsets=np.concatenate(all_subsets), edges=np.concatenate(all_edges), num_nodes=np.concatenate(all_num_nodes), num_edges=np.concatenate(all_num_edges))

    @staticmethod
    def batch_radius_graph(batch_pos, L, max_edges):
        """
        Batched geometry (batch_size, n_atoms, 3) -> indices of edges within L (batch_size, n_edges, 2), number of edges within L (batch_size, )
        """
        def _distance_matrix(pos):
            """
            Geometry (n_atoms, 3) -> pairwise distances (n_atoms, n_atoms)
            Batched geometry (batch_size, n_atoms, 3) -> pairwise distances (batch_size, n_atoms, n_atoms)
            """
            assert(len(pos.shape) == 2 or len(pos.shape) == 3)
            return onp.linalg.norm(onp.expand_dims(pos, -2) - onp.expand_dims(pos, -3), axis=-1)

        def _radius_graph(pos, L):
            """
            Geometry (n_atoms, 3) -> indices of edges within L (n_edges, 2)
            """
            assert(len(pos.shape) == 2)
            return onp.argwhere((_distance_matrix(pos) < L) & ~onp.identity(pos.shape[-2], dtype=bool))

        assert(len(batch_pos.shape) == 3)
        all_edges = []    
        all_num_edges = []
        for i, pos in enumerate(batch_pos):
            edges = _radius_graph(pos, L)
            num_edges = len(edges)
            pad_len = max_edges - num_edges
            if pad_len < 0:
                print(f"Skipping {i}")
                continue
            all_edges.append(onp.pad(edges, ((0, pad_len), (0, 0)), mode="constant", constant_values=-1))
            all_num_edges.append(num_edges)
        return onp.array(all_edges, dtype=onp.int32), onp.array(all_num_edges, dtype=onp.int16)

if __name__ == "__main__": 
	spice_serializer = SPICESerializer('SPICE-1.1.3.hdf5', sys.argv[1], train_ratio=float(sys.argv[2]), test_ratio=float(sys.argv[3]), max_atoms=int(sys.argv[4]))
