import h5py
import jax
import numpy as np
import time
import json
import os
import sys

class SPICESerializer:

    def __init__(self, data_path, out_prefix, train_ratio, test_ratio, seed=2666, max_atom_num=96):
        self.data = h5py.File(data_path, 'r')
        self.names = list(self.data.keys())
        key = jax.random.PRNGKey(seed)
        self.max_atom_num = max_atom_num
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
        all_forces = []
        for name in names:
            atom_nums = np.array(self.data[name]['atomic_numbers'], np.uint8)
            if len(atom_nums) > self.max_atom_num:
                print("Skipping: ", name)
                continue
            pos_arr = self.data[name]['conformations']
            forces_arr = -self.data[name]['dft_total_gradient']
            # total_energy_arr = self.data[name]['dft_total_energy']
            form_energy_arr = self.data[name]['formation_energy']
            pad_num = self.max_atom_num - len(atom_nums)
            padded_atom_nums = np.pad(atom_nums, (0, pad_num))
            padded_pos = np.pad(pos_arr, ((0, 0), (0, pad_num), (0, 0)))
            padded_forces = np.pad(forces_arr, ((0, 0), (0, pad_num), (0, 0)))
            all_atom_nums.append([padded_atom_nums for conf in range(len(pos_arr))])
            # all_total_energies.append(total_energy_arr)
            all_form_energies.append(form_energy_arr)
            all_forces.append(padded_forces)
            all_pos.append(padded_pos)
        np.savez(out_path, atomic_numbers=np.concatenate(all_atom_nums), formation_energy=np.concatenate(all_form_energies), forces=np.concatenate(all_forces), pos=np.concatenate(all_pos))

if __name__ == "__main__": 
	spice_serializer = SPICESerializer('SPICE-1.1.3.hdf5', sys.argv[1], train_ratio=float(sys.argv[2]), test_ratio=float(sys.argv[3]), max_atom_num=int(sys.argv[4]))
