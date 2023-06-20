import h5py
import jax
import numpy as np
import time
import json
import os

class SPICESerializer:

    def __init__(self, data_path, train_ratio, test_ratio, seed=2666, max_atom_num=96):
        self.data = h5py.File(data_path)
        self.names = list(self.data.keys())
        key = jax.random.PRNGKey(seed)
        self.max_atom_num = max_atom_num
        self.test_names, self.train_names, self.val_names = self._split(key, train_ratio, test_ratio)
        self._make_npz(self.train_names, "spice_train")
        self._make_npz(self.test_names, "spice_test")
        # self.make_npz(self.val_names, "spice_val")

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
        all_energies = []
        all_forces = []
        for name in names:
            atom_nums = np.array(self.data[name]['atomic_numbers'], np.uint8)
            if len(atom_nums) > self.max_atom_num:
                print("Skipping: ", name)
                continue
            pos_arr = self.data[name]['conformations']
            forces_arr = self.data[name]['dft_total_gradient']
            energy_arr = self.data[name]['dft_total_energy']
            pad_num = self.max_atom_num - len(atom_nums)
            padded_atom_nums = np.pad(atom_nums, (0, pad_num))
            padded_pos = np.pad(pos_arr, ((0, 0), (0, pad_num), (0, 0)))
            padded_forces = np.pad(forces_arr, ((0, 0), (0, pad_num), (0, 0)))
            all_atom_nums.append([padded_atom_nums for conf in range(len(pos_arr))])
            all_energies.append(energy_arr)
            all_forces.append(padded_forces)
            all_pos.append(padded_pos)
        np.savez(out_path, atomic_numbers=np.concatenate(all_atom_nums), total_energy=np.concatenate(all_energies), forces=np.concatenate(all_forces), pos=np.concatenate(all_pos))


class SpiceLoader:

    def __init__(self, file_prefix, seed=2666, batch_size=128, mmap_mode=None):
        self.key = jax.random.PRNGKey(seed)
        self.batch_size = batch_size
        self.train_energies = np.load(file_prefix + "_total_energy.npy", mmap_mode=mmap_mode)
        self.train_atom_nums = np.load(file_prefix + "_atomic_numbers.npy", mmap_mode=mmap_mode)
        self.train_pos = np.load(file_prefix + "_pos.npy")

    def __getitem__(self, i):
        return self.train_atom_nums[i], self.train_pos[i], self.train_energies[i]

    def __len__(self):
        return len(self.train_energies)

    def get_epoch(self):
        train_size = len(self.train_energies)
        shuffled_idxs = jax.random.permutation(self.key, train_size)
        for batch_num in range(train_size // self.batch_size):
            batch_start = batch_num * self.batch_size
            batch_end = batch_start + self.batch_size
            batch_idxs = shuffled_idxs[batch_start:batch_end]
            batch_atom_nums = self.train_atom_nums[batch_idxs]
            batch_pos = self.train_pos[batch_idxs]
            batch_energies = self.train_energies[batch_idxs]
            yield batch_atom_nums, batch_pos, batch_energies


spice_serializer = SPICESerializer('SPICE-1.1.3.hdf5', train_ratio=0.001, test_ratio=0.001, max_atom_num=60)
# spice_loader = SpiceLoader(mmap_mode="r")
# start = time.time()

# for batch in spice_loader.get_epoch():
#     atom_nums, pos, energies = batch

# end = time.time()
# print(end - start)
